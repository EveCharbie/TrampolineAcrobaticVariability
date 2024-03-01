from Function_Class_Basics import create_composite_image, add_lines_with_arrow_and_circle, graph_images_info, lines_info
from PIL import Image

# https://www.shutterstock.com/fr/search/corps-humain-de-dos


file_path = "/home/lim/Documents/StageMathieu/Graph_from_mot/SaMi/MeanSD/"
folder_path = "/home/lim/Documents/StageMathieu/Graph_from_mot/SaMi/Graph_with_body.png"

# Call the function to create the composite image
composite_image_path = create_composite_image(graph_images_info, file_path, save_path=folder_path)


# Call the function
output_image_path = add_lines_with_arrow_and_circle(composite_image_path, lines_info)
