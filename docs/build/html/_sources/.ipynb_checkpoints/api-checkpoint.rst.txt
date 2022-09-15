ImpressionismCat API
********************

Neural Style Transfer
=====================

.. autoclass:: impressionismcat.paint.StyleTransfer


Preprocess
----------

.. autofunction:: impressionismcat.paint.StyleTransfer.load_resize_img

.. autofunction:: impressionismcat.paint.StyleTransfer.process_img_as_vgg_input

.. autofunction:: impressionismcat.paint.StyleTransfer.deprocess_img

.. autofunction:: impressionismcat.paint.StyleTransfer.represent_features

Neural Network Architecture
---------------------------

.. autofunction:: impressionismcat.paint.StyleTransfer.setup_model


Loss Function
-------------

.. autofunction:: impressionismcat.paint.StyleTransfer.gram_matrix

.. autofunction:: impressionismcat.paint.StyleTransfer.get_loss_content

.. autofunction:: impressionismcat.paint.StyleTransfer.get_loss_style

.. autofunction:: impressionismcat.paint.StyleTransfer.compute_loss

Optimization
------------

.. autofunction:: impressionismcat.paint.StyleTransfer.compute_grads

.. autofunction:: impressionismcat.paint.StyleTransfer.optimize

Display
-------

.. autofunction:: impressionismcat.paint.StyleTransfer.show_inputs

.. autofunction:: impressionismcat.paint.StyleTransfer.show_results

.. autofunction:: impressionismcat.paint.StyleTransfer.save_gif

.. autofunction:: impressionismcat.paint.StyleTransfer.save_pic