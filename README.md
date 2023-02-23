# SUPERVISELY_TEST

## TASK 1
* The script accepts a path to image (`-p`), size (height, width) of the window in pixels as "100x100"(`-s`), 
offset on x and y (`-x`, `-y`), and slices image sliding window approach.
* The result is saved as .png, files because when saving to jpg, information is lost when the file is compressed .
* All settings are saved in the name of the folder with the output pictures.
* #### Example:
<!-- end of the list -->
    cd task1

    # splitting image
    python split_image.py -p image_name.jpg -s 200x200 -x 100 -y 100

    # merging image
     python merge_image.py -p result_split/image_name_600_800_3_200x200_x100_y100 -o image_name.jpg 

## TASK 2
* Visualizer for dataset [DAVIS](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip).
* The script accepts a path to folder dataset(`-p`) with cut pictures as input and collects the original image from them.
* Finally, it verifies that the pixels of the result image are exactly the same as the original image.
* [overlay mask on pictures](https://github.com/albertomontesg/davis-interactive/blob/master/davisinteractive/utils/visualization.py)
* [create video in the form of a grid](https://gist.github.com/luuil/183e4d92275e3d6641b6728a988bb38b)
<!-- end of the list -->
    cd task2
    
    # run visualisator for create line video
    python data_visualisator.py -p DAVIS
    
    # run visualisator for create grid video
    python data_visualisator.py -p DAVIS -g True
## TASK 3
### TO DO