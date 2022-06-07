import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False

import numpy as np
from PIL import Image
import time
import functools
import tensorflow as tf

def load_resize_img(path, max_dim=512):
    img = Image.open(path)
    
    long = max(img.size)
    scale = max_dim/long
    
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    # We need to broadcast the image array such that it has a batch dimension 
    img_array = np.expand_dims(img, axis=0)
    return img_array

def display_img(img_array, title=None):
    # Remove the batch dimension
    out = np.squeeze(img_array, axis=0)
    # Normalize for display 
    out = out.astype('uint8')
    plt.imshow(out)
    if title:
        plt.title(title)
    return None


## algorithm

### input

def load_img_as_vgg_input(path):
    img_array = load_resize_img(path)
    img_vgg = tf.keras.applications.vgg19.preprocess_input(img_array)
    return img_vgg

def deprocess_img(img_processed):
    x = img_processed.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # perform the inverse of the preprocessing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


### model

def setup_model():
    # Content layer where will pull our feature maps
    layers_content = ['block5_conv2'] 

    # Style layer we are interested in
    layers_style = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1']

    global num_layers_content
    global num_layers_style
    num_layers_content = len(layers_content)
    num_layers_style = len(layers_style)

    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    # Get output layers corresponding to style and content layers 
    outputs_style = [vgg.get_layer(name).output for name in layers_style]
    outputs_content = [vgg.get_layer(name).output for name in layers_content]
    outputs_model = outputs_style + outputs_content

    model = tf.keras.Model(inputs=vgg.input, outputs=outputs_model, name='style transfer')
    for layer in model.layers:
        layer.trainable = False
    return model


### representitions

def feature_representations(model, path_content, path_style):
    # Load our images in 
    input_vgg_content = load_img_as_vgg_input(path_content)
    input_vgg_style = load_img_as_vgg_input(path_style)

    # batch compute content and style features
    outputs_style = model(input_vgg_style)
    outputs_content = model(input_vgg_content)

    # Get the style and content feature representations from our model  
    features_style = [style_layer[0] for style_layer in outputs_style[:num_layers_style]]
    features_content = [content_layer[0] for content_layer in outputs_content[num_layers_style:]]
    return features_style, features_content


### loss function

def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

def get_style_loss(base_style, gram_target):
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))

def compute_loss(model, loss_weights, input_vgg_init, features_style_gram, features_content):

    weight_style, weight_content = loss_weights

    # Feed our init image through our model. This will give us the content and 
    # style representations at our desired layers. Since we're using eager
    # our model is callable just like any other function!
    outputs_model = model(input_vgg_init)

    style_output_features = outputs_model[:num_layers_style]
    content_output_features = outputs_model[num_layers_style:]

    score_style = 0
    score_content = 0

    # Accumulate style losses from all layers
    # Here, we equally weight each contribution of each loss layer
    weight_per_style_layer = 1.0 / float(num_layers_style)
    for target_style, comb_style in zip(features_style_gram, style_output_features):
        score_style += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    # Accumulate content losses from all layers 
    weight_per_content_layer = 1.0 / float(num_layers_content)
    for target_content, comb_content in zip(features_content, content_output_features):
        score_content += weight_per_content_layer* get_content_loss(comb_content[0], target_content)

    score_style *= weight_style
    score_content *= weight_content

    # Get total loss
    loss_total = score_style + score_content 
    return loss_total, score_style, score_content

## optimization

def compute_grads(cfg):
    with tf.GradientTape() as tape: 
        losses = compute_loss(**cfg)
        
    # Compute gradients wrt input image
    loss_total = losses[0]
    return tape.gradient(loss_total, cfg['input_vgg_init']), losses


def optimize(best_img, best_loss, input_vgg_init, 
             iterations=1000, cfg=None, 
             display=True, display_interval=1):
    
    # Create our optimizer
    opt = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)
    
    start_time = time.time()
    global_start = time.time()

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means 
    
    imgs = []
    for i in range(iterations):
        grads, losses = compute_grads(cfg)
        loss, score_style, score_content = losses
        opt.apply_gradients([(grads, input_vgg_init)])
        clipped = tf.clip_by_value(input_vgg_init, min_vals, max_vals)
        input_vgg_init.assign(clipped)
        end_time = time.time() 

        if loss < best_loss:
            # Update best loss and best image from total loss. 
            best_loss = loss
            best_img = deprocess_img(input_vgg_init.numpy())
        
        if display:
            import IPython.display
            if i % display_interval== 0:
                start_time = time.time()

                # Use the .numpy() method to get the concrete numpy array
                plot_img = input_vgg_init.numpy()
                plot_img = deprocess_img(plot_img)
                imgs.append(plot_img)
                IPython.display.clear_output(wait=True)
                IPython.display.display_png(Image.fromarray(plot_img))
                print('Iteration: {}'.format(i))        
                print('Total loss: {:.2e}, ' 
                      'style loss: {:.2e}, '
                      'content loss: {:.2e}, '
                      'time: {:.4f}s'.format(loss, score_style, score_content, time.time() - start_time))
    return best_img, best_loss, input_vgg_init

def styleTransfer(path_content=None, path_style=None, iterations=1000, 
                  display=False, display_interval=1, continue_transfer=False):
    
    model = setup_model()
    
    # Get the style and content feature representations (from our specified intermediate layers) 
    features_style, features_content = feature_representations(model, path_content, path_style)
    features_style_gram = [gram_matrix(feature_style) for feature_style in features_style]

    # Set initial image
    input_vgg_init = load_img_as_vgg_input(path_content)
    input_vgg_init = tf.Variable(input_vgg_init, dtype=tf.float32)
    
    weight_content=1e3
    weight_style=1e-2
    
    loss_weights = (weight_style, weight_content)

    cfg = {
      'model': model,
      'loss_weights': loss_weights,
      'input_vgg_init': input_vgg_init,
      'features_style_gram': features_style_gram,
      'features_content': features_content
    }
    
    best_loss, best_img = float('inf'), None
    
    best_img, best_loss, input_vgg_init = optimize(best_img, best_loss, input_vgg_init,
                                                   iterations=1000, cfg=cfg, 
                                                   display=True, display_interval=1)
    if continue_transfer:
        return best_img, best_loss, cfg
    
    Image.fromarray(best_img)
    return best_img
    

def show_results(best_img, path_content, path_style, show_large_final=True):
    plt.figure(figsize=(10, 5))
    content = load_resize_img(path_content) 
    style = load_resize_img(path_style)

    plt.subplot(1, 2, 1)
    display_img(content, 'Content Image')

    plt.subplot(1, 2, 2)
    display_img(style, 'Style Image')

    if show_large_final: 
        plt.figure(figsize=(10, 10))

        plt.imshow(best_img)
        plt.title('Output Image')
        plt.show()
    return None

def show_inputs(path_content, path_style):
    plt.figure(figsize=(15,15))

    content = load_resize_img(path_content).astype('uint8')
    style = load_resize_img(path_style).astype('uint8')

    plt.subplot(1, 2, 1)
    display_img(content, 'Content Image')

    plt.subplot(1, 2, 2)
    display_img(style, 'Style Image')
    plt.show()
    return None