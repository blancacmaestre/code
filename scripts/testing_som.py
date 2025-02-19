import fitz  # PyMuPDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image
import io
import os

# Step 1: Extract images from the source PDF
def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    image_list = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list.extend(page.get_images(full=True))

    images = []
    for img in image_list:
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        images.append(image_bytes)
    return images

# Step 2: Create a table in the new PDF and insert the images
def create_pdf_with_images(images, output_pdf_path):
    c = canvas.Canvas(output_pdf_path, pagesize=letter)
    width, height = letter
    rows = 5  # Set the number of rows in the table
    cols = 4  # Set the number of columns in the table
    cell_width = width / cols
    cell_height = height / rows

    # Create a table structure
    image_index = 0
    for row in range(rows):
        for col in range(cols):
            if image_index < len(images):
                # Save the image to a temporary file
                img = Image.open(io.BytesIO(images[image_index]))
                img_path = f"temp_image_{image_index}.png"
                img.save(img_path)
                
                # Place the image into the PDF
                c.drawImage(img_path, col * cell_width, height - (row + 1) * cell_height, width=cell_width, height=cell_height)
                os.remove(img_path)  # Clean up the temporary image file
                image_index += 1

    c.save()

# Main function to run the script
def main():
    # Set the paths for input PDF (you need to adjust the paths to your files)
    pdf_paths = ["/home/user/THESIS/models/A_MODELS_new/new_attempt/CGal_4_30_0.01/CGal_4_30_0.01.fits","/home/user/THESIS/models/A_MODELS_new/new_attempt/CGal_4_40_0.01/CGal_4_40_0.01.fits","/home/user/THESIS/models/A_MODELS_new/new_attempt/CGal_4_50_0.01/CGal_4_50_0.01.fits","/home/user/THESIS/models/A_MODELS_new/new_attempt/CGal_4_60_0.01/CGal_4_60_0.01.fits"] 
    output_pdf_path = "/home/user/THESIS/output_with_table.pdf"  # Path for the new PDF

    all_images = []

    # Step 1: Extract images from all PDF files
    for pdf_path in pdf_paths:
        print(f"Extracting images from: {pdf_path}")
        images = extract_images_from_pdf(pdf_path)
        all_images.extend(images)

    # Step 2: Create PDF with images inserted in a table
    print(f"Creating PDF with {len(all_images)} images...")
    create_pdf_with_images(all_images, output_pdf_path)
    print(f"PDF with table and images saved to: {output_pdf_path}")

if __name__ == "__main__":
    main()