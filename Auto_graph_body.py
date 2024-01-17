from PIL import Image, ImageDraw
import math
# https://www.shutterstock.com/fr/search/corps-humain-de-dos


# Define a function to create a composite image with the body at the center
def create_composite_image_with_borders(graph_images_info, bg_size=(1920, 1082), body_size=(383, 669), body_position=(761, 228), graph_size=(366, 220), border_thickness=0):
    """
    Create a composite image with a central body image, surrounding graph images, and a border around each graph.

    :param graph_images_info: A dictionary with graph image filenames as keys and positions as values.
    :param bg_size: Size of the background image (width, height).
    :param body_size: Size to resize the body image (width, height).
    :param body_position: Position to place the body image (x, y).
    :param graph_size: Size to resize each graph image (width, height).
    :param border_thickness: Thickness of the border around each graph.
    :return: Path to the saved composite image.
    """
    # Create a blank white background
    background = Image.new('RGB', bg_size, color='white')

    # Load and place the body image
    body_image_path = '/home/lim/Documents/StageMathieu/Graph_from_mot/SaMi/DALL_E_Body.png'
    try:
        body_image = Image.open(body_image_path)
        body_image = body_image.resize(body_size, Image.Resampling.LANCZOS)
        background.paste(body_image, body_position, body_image)
    except FileNotFoundError:
        print("Body image file not found. Please upload the file and try again.")
        return None

    # Load, resize, and place each graph image with a border
    for graph_image_name, position in graph_images_info.items():
        graph_image_path = f'/home/lim/Documents/StageMathieu/Graph_from_mot/SaMi/MeanSD/{graph_image_name}'
        try:
            # Open and resize the graph image
            graph_image = Image.open(graph_image_path)
            graph_image = graph_image.resize(graph_size, Image.Resampling.LANCZOS)

            # Create a border for the graph image
            bordered_image = Image.new('RGB', (graph_size[0] + 2*border_thickness, graph_size[1] + 2*border_thickness), color='black')
            bordered_image.paste(graph_image, (border_thickness, border_thickness))

            # Place the bordered image on the background
            background.paste(bordered_image, (position[0] - border_thickness, position[1] - border_thickness))
        except FileNotFoundError:
            print(f"Graph image file {graph_image_name} not found. Skipping this image.")

    # Save the composite image
    composite_image_path = '/home/lim/Documents/StageMathieu/Graph_from_mot/SaMi/composite_image_no_borders.png'
    background.save(composite_image_path)
    return composite_image_path


graph_images_info = {
    'AvBrasG_all_axes_graph.png': (1212, 327),
    'AvBrasD_all_axes_graph.png': (358, 327),
    'BrasG_all_axes_graph.png': (1544, 186),
    'BrasD_all_axes_graph.png': (29, 186),
    'CuisseG_all_axes_graph.png': (1211, 558),
    'CuisseD_all_axes_graph.png': (358, 558),
    'EpauleG_all_axes_graph.png': (1298, -10),  # Note: Negative values might need special handling
    'EpauleD_all_axes_graph.png': (285, -10),
    'MainG_all_axes_graph.png': (1554, 443),
    'MainD_all_axes_graph.png': (16, 443),
    'JambeG_all_axes_graph.png': (1544, 804),
    'JambeD_all_axes_graph.png': (53, 804),
    'PiedG_all_axes_graph.png': (1153, 865),
    'PiedD_all_axes_graph.png': (443, 865),
    'Pelvis_all_axes_graph.png': (793, 865),
    'Thorax_all_axes_graph.png': (628, -10),
    'Tete_all_axes_graph.png': (963, -10)
}



# Call the function to create the composite image with borders around the graphs
composite_image_path_with_borders = create_composite_image_with_borders(graph_images_info)
composite_image_path_with_borders


def add_lines_with_arrow_and_circle(image_path, lines_info, line_width=2, arrow_size=15, scale_factor=4):
    """
    Draw smooth lines with arrows on the image using a scaling technique for anti-aliasing.

    :param image_path: Path to the image where lines will be drawn.
    :param lines_info: A dictionary with keys as line identifiers and values as tuples containing
                       start and end coordinates (x1, y1, x2, y2).
    :param line_width: Width of the lines.
    :param arrow_size: Size of the arrow.
    :param scale_factor: Factor by which to scale the image for drawing.
    :return: Path to the saved image with smooth drawn lines.
    """
    # Load the image and scale it up
    with Image.open(image_path) as img:
        large_img = img.resize((img.width * scale_factor, img.height * scale_factor), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(large_img)

        for start, end in lines_info.values():
            # Scale up coordinates
            start = tuple([x * scale_factor for x in start])
            end = tuple([x * scale_factor for x in end])
            scaled_line_width = line_width * scale_factor
            scaled_arrow_size = arrow_size * scale_factor

            # Draw the line
            draw.line((start, end), fill="black", width=scaled_line_width)

            # Calculate arrow direction
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            angle = math.atan2(dy, dx)

            # Calculate arrow points
            arrow_tip = end
            arrow_left = (end[0] - scaled_arrow_size * math.cos(angle - math.pi/6), end[1] - scaled_arrow_size * math.sin(angle - math.pi/6))
            arrow_right = (end[0] - scaled_arrow_size * math.cos(angle + math.pi/6), end[1] - scaled_arrow_size * math.sin(angle + math.pi/6))

            # Draw the arrow
            draw.polygon([arrow_tip, arrow_left, arrow_right], fill="black")

        # Resize the image back down with anti-aliasing
        smooth_img = large_img.resize(img.size, Image.Resampling.LANCZOS)

        # Save the modified image
        output_path = image_path.replace(".png", "_smooth_lines.png")
        smooth_img.save(output_path)

    return output_path
# Example usage:
lines_info = {
    'line1': ((952, 400), (1130, 210)),  # Replace with actual coordinates
    'line2': ((952, 497), (794, 210)),  # Replace with actual coordinates
    # Add more lines as needed
}

# Call the function
output_image_path = add_lines_with_arrow_and_circle("/home/lim/Documents/StageMathieu/Graph_from_mot/SaMi/composite_image_no_borders.png", lines_info)
