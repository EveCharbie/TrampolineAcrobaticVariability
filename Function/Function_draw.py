from PIL import Image, ImageDraw
import math


lines_info = {
    "line1": ((952, 400), (1130, 210)),  # Replace with actual coordinates
    "line2": ((952, 450), (794, 210)),  # Replace with actual coordinates
    "line3": ((935, 411), (570, 190)),  # Replace with actual coordinates
    "line4": ((903, 424), (370, 250)),  # Replace with actual coordinates
    "line5": ((898, 496), (720, 436)),  # Replace with actual coordinates
    "line6": ((879, 557), (360, 553)),  # Replace with actual coordinates
    "line7": ((929, 546), (720, 667)),  # Replace with actual coordinates
    "line8": ((929, 649), (400, 890)),  # Replace with actual coordinates
    "line9": ((930, 762), (609, 880)),  # Replace with actual coordinates
    "line10": ((953, 532), (960, 880)),  # Replace with actual coordinates
    "line11": ((976, 762), (1319, 880)),  # Replace with actual coordinates
    "line12": ((976, 649), (1543, 890)),  # Replace with actual coordinates
    "line13": ((976, 546), (1221, 667)),  # Replace with actual coordinates
    "line14": ((1024, 557), (1551, 553)),  # Replace with actual coordinates
    "line15": ((1008, 496), (1222, 436)),  # Replace with actual coordinates
    "line16": ((1000, 424), (1543, 250)),  # Replace with actual coordinates
    "line17": ((971, 411), (1410, 190)),  # Replace with actual coordinates
}

graph_images_info = {
    "Thorax_all_axes_graph.png": (623, -10),
    "Tete_all_axes_graph.png": (960, -10),
    "CuisseG_all_axes_graph.png": (1211, 558),
    "CuisseD_all_axes_graph.png": (358, 558),
    "EpauleG_all_axes_graph.png": (1298, -10),
    "EpauleD_all_axes_graph.png": (285, -10),
    "BrasG_all_axes_graph.png": (1554, 186),
    "BrasD_all_axes_graph.png": (16, 186),
    "AvBrasG_all_axes_graph.png": (1212, 340),
    "AvBrasD_all_axes_graph.png": (358, 340),
    "MainG_all_axes_graph.png": (1554, 443),
    "MainD_all_axes_graph.png": (16, 443),
    "JambeG_all_axes_graph.png": (1544, 804),
    "JambeD_all_axes_graph.png": (53, 804),
    "PiedG_all_axes_graph.png": (1153, 865),
    "PiedD_all_axes_graph.png": (443, 865),
    "Pelvis_all_axes_graph.png": (793, 865),
}


column_names = [
    "PelvisTranslation_X",
    "PelvisTranslation_Y",
    "PelvisTranslation_Z",
    "Pelvis_X",
    "Pelvis_Y",
    "Pelvis_Z",
    "Thorax_X",
    "Thorax_Y",
    "Thorax_Z",
    "Tete_X",
    "Tete_Y",
    "Tete_Z",
    "EpauleD_Y",
    "EpauleD_Z",
    "BrasD_X",
    "BrasD_Y",
    "BrasD_Z",
    "AvBrasD_X",
    "AvBrasD_Z",
    "MainD_X",
    "MainD_Y",
    "EpauleG_Y",
    "EpauleG_Z",
    "BrasG_X",
    "BrasG_Y",
    "BrasG_Z",
    "AvBrasG_X",
    "AvBrasG_Z",
    "MainG_X",
    "MainG_Y",
    "CuisseD_X",
    "CuisseD_Y",
    "CuisseD_Z",
    "JambeD_X",
    "PiedD_X",
    "PiedD_Z",
    "CuisseG_X",
    "CuisseG_Y",
    "CuisseG_Z",
    "JambeG_X",
    "PiedG_X",
    "PiedG_Z",
]


def dessiner_vecteurs(ax, origine, vecteur_x, vecteur_y, vecteur_z, longueur=0.1):
    ax.quiver(
        origine[0],
        origine[1],
        origine[2],
        vecteur_x[0],
        vecteur_x[1],
        vecteur_x[2],
        color="r",
        length=longueur,
        normalize=True,
    )
    ax.quiver(
        origine[0],
        origine[1],
        origine[2],
        vecteur_y[0],
        vecteur_y[1],
        vecteur_y[2],
        color="g",
        length=longueur,
        normalize=True,
    )
    ax.quiver(
        origine[0],
        origine[1],
        origine[2],
        vecteur_z[0],
        vecteur_z[1],
        vecteur_z[2],
        color="b",
        length=longueur,
        normalize=True,
    )


def create_composite_image(
    graph_images_info,
    base_graph_path,
    save_path,
    bg_size=(1920, 1082),
    body_size=(383, 669),
    body_position=(761, 228),
    graph_size=(366, 220),
    border_thickness=0,
):
    background = Image.new("RGB", bg_size, color="white")

    body_image_path = "/Graph_from_mot/DALL_E_Body.png"
    try:
        body_image = Image.open(body_image_path)
        body_image = body_image.resize(body_size, Image.Resampling.LANCZOS)
        background.paste(body_image, body_position, body_image)
    except FileNotFoundError:
        print("Body image file not found. Please upload the file and try again.")
        return None

    full_graph_images_info = {
        base_graph_path + filename: position
        for filename, position in graph_images_info.items()
    }

    for graph_image_filename, graph_position in full_graph_images_info.items():
        try:
            graph_image = Image.open(graph_image_filename)
            graph_image = graph_image.resize(graph_size, Image.Resampling.LANCZOS)

            if border_thickness > 0:
                border_image = Image.new(
                    "RGB",
                    (
                        graph_size[0] + 2 * border_thickness,
                        graph_size[1] + 2 * border_thickness,
                    ),
                    color="black",
                )
                border_position = (
                    graph_position[0] - border_thickness,
                    graph_position[1] - border_thickness,
                )
                background.paste(border_image, border_position)
                background.paste(graph_image, graph_position, graph_image)
            else:
                background.paste(graph_image, graph_position, graph_image)
        except FileNotFoundError:
            print(f"Graph image file {graph_image_filename} not found.")
    legend_image_path = base_graph_path + "legend.png"
    try:
        legend_image = Image.open(legend_image_path)
        background.paste(legend_image, (1723, 0))
    except FileNotFoundError:
        print("Legend image file not found")
    try:
        background.save(save_path)
    except Exception as e:
        print(f"Error saving the image: {e}")
        return None
    return save_path


def add_lines_with_arrow_and_circle(
    image_path, lines_info, line_width=2, arrow_size=15, circle_radius=5, scale_factor=4
):
    """
    Draw smooth lines with arrows on one end and circles on the other on the image using a scaling technique for
    anti-aliasing.

    :param image_path: Path to the image where lines will be drawn.
    :param lines_info: A dictionary with keys as line identifiers and values as tuples containing
                       start and end coordinates (x1, y1, x2, y2).
    :param line_width: Width of the lines.
    :param arrow_size: Size of the arrow.
    :param circle_radius: Radius of the circle.
    :param scale_factor: Factor by which to scale the image for drawing.
    :return: Path to the saved image with smooth drawn lines and circles.
    """
    # Load the image and scale it up
    with Image.open(image_path) as img:
        large_img = img.resize(
            (img.width * scale_factor, img.height * scale_factor),
            Image.Resampling.LANCZOS,
        )
        draw = ImageDraw.Draw(large_img)

        for start, end in lines_info.values():
            # Scale up coordinates and dimensions
            start_scaled = tuple([x * scale_factor for x in start])
            end_scaled = tuple([x * scale_factor for x in end])
            line_width_scaled = line_width * scale_factor
            arrow_size_scaled = arrow_size * scale_factor
            circle_radius_scaled = circle_radius * scale_factor

            # Draw the line
            draw.line((start_scaled, end_scaled), fill="black", width=line_width_scaled)

            # Calculate arrow direction
            dx = end_scaled[0] - start_scaled[0]
            dy = end_scaled[1] - start_scaled[1]
            angle = math.atan2(dy, dx)

            # Calculate arrow points
            arrow_tip = end_scaled
            arrow_left = (
                end_scaled[0] - arrow_size_scaled * math.cos(angle - math.pi / 6),
                end_scaled[1] - arrow_size_scaled * math.sin(angle - math.pi / 6),
            )
            arrow_right = (
                end_scaled[0] - arrow_size_scaled * math.cos(angle + math.pi / 6),
                end_scaled[1] - arrow_size_scaled * math.sin(angle + math.pi / 6),
            )

            # Draw the arrow
            draw.polygon([arrow_tip, arrow_left, arrow_right], fill="black")

            # Draw a filled circle
            draw.ellipse(
                [
                    (
                        start_scaled[0] - circle_radius_scaled,
                        start_scaled[1] - circle_radius_scaled,
                    ),
                    (
                        start_scaled[0] + circle_radius_scaled,
                        start_scaled[1] + circle_radius_scaled,
                    ),
                ],
                fill="black",
            )

        # Resize the image back down with anti-aliasing
        smooth_img = large_img.resize(img.size, Image.Resampling.LANCZOS)

        # Save the modified image
        output_path = image_path.replace(".png", ".png")
        smooth_img.save(output_path)

    return output_path


def plot_adjusted_fd(ax, mean_fd, adjusted_fd_positive, adjusted_fd_negative, title, grid_points,
                     percentage_variabilite):
    ax.plot(grid_points, mean_fd.data_matrix[0, :, 0], 'black', label='Moyenne', linewidth=2)
    ax.plot(grid_points, adjusted_fd_positive.data_matrix[0, :, 0], 'g+', linestyle='none', label='Moyenne + fPC',
            markersize=5, linewidth=2)
    ax.plot(grid_points, adjusted_fd_negative.data_matrix[0, :, 0], 'r_', linestyle='none', label='Moyenne - fPC',
            markersize=5, linewidth=2)
    ax.set_title(f"{title} qui explique {percentage_variabilite} de la variabilite")
    ax.set_xlabel('Temps')
    ax.set_ylabel('Vitesse Angulaire')
    ax.legend()


