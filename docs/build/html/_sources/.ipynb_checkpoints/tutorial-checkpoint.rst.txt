ImpressionismCat Tutorials
**************************

Neural Style Transfer
=====================

First, import ImpressionismCat to your workspace like Jupyter Notebook. To do a neural style transfer art, you need two images at least:

1. **A content image**, some you would like to present as the content.
2. **A style image**, such as an artwork by a famous artist.

In the tutorial, we use a **Volvo C40** as the content image and **The Great Wave off Kanagawa** as the style image. 

.. note::

    1. The Volvo C40 is an battery electric subcompact luxury crossover SUV with a sloping roofline manufactured by Volvo Cars, which was released on 2 March 2021. It is also the first Volvo model that is only available as a battery electric vehicle.

    2. The Great Wave off Kanagawa (Japanese: 神奈川沖浪裏, Hepburn: Kanagawa-oki Nami Ura, lit. "Under the Wave off Kanagawa") is a woodblock print that was made by Japanese ukiyo-e artist Hokusai, probably in late 1831 during the Edo period of Japanese history. The print depicts three boats moving through a storm-tossed sea with a large wave forming a spiral in the centre and Mount Fuji visible in the background. 

.. code-block:: Python

    import impressionismcat as ic
    
    path_content = 'img/c40_recharge.jpg'
    path_style = 'img/great_wave.jpg'

Secondly, we initialize a :code:`StyleTransfer` Class as :code:`style`. Before transfering the style, we check the inputs. 

.. code-block:: Python

    style = ic.paint.StyleTransfer(path_content, path_style)
    
    style.show_inputs()

.. image:: images/inputs.png
  :align: center

Now it's time to do the transfer! We set 2000 iterations and output the result.

.. code-block:: Python

    style.optimize(iterations=2000)
    
    style.show_results()

.. image:: images/result.png
  :align: center

ImpressionismCat provides function to generate a gif to record the transfer progress. Make it fun!

.. code-block:: Python

    style.save_gif('style_greatWave.gif')

.. image:: images/style_greatwave.gif
  :align: center

