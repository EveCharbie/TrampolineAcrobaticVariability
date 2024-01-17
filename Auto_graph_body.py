from PIL import Image



# https://www.shutterstock.com/fr/search/corps-humain-de-dos


# Create a blank white image with the dimensions 1920x1082
bg_size = (1920, 1082)
background = Image.new('RGB', bg_size, color='white')

# Load the body image (assuming it's named 'body_image.png')
# Here, we assume the body image has been uploaded and is named 'body_image.png' in the /mnt/data/ directory.
body_image_path = '/home/lim/Documents/StageMathieu/Graph_from_mot/SaMi/DALL_E_Body.png'  # Replace with the correct path if necessary
try:
    # Attempt to open the body image
    body_image = Image.open(body_image_path)

    # Resize the body image to 382.5x669.3
    # The size must be integers in PIL, so we round the dimensions
    body_size = (383, 669)  # Rounded dimensions
    body_image = body_image.resize(body_size, Image.Resampling.LANCZOS)

    # Set the position where the body image will be placed on the background
    body_position = (761, 228)  # Rounded coordinates
    background.paste(body_image, body_position)

    # Save the composite image
    composite_image_path = '/home/lim/Documents/StageMathieu/Graph_from_mot/SaMi/composite_image.png'
    background.save(composite_image_path)

    # Display the path to the saved image
    composite_image_path
except FileNotFoundError:
    print("The body image file was not found. Please upload the file and try again.")
