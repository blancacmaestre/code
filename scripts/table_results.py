import os
import cv2
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image

def extract_graphs_from_pdf(pdf_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    images = convert_from_path(pdf_path)
    extracted_graphs = []
    
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f"page_{i+1}.png")
        image.save(image_path, "PNG")
        
        img_cv = cv2.imread(image_path)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found_graphs = False
        for j, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 50:  # Adjust size threshold if needed
                graph = img_cv[y:y+h, x:x+w]
                graph_path = os.path.join(output_folder, f"graph_{i+1}_{j+1}.png")
                cv2.imwrite(graph_path, graph)
                extracted_graphs.append((pdf_path, i+1, graph_path, f"Graph_{i+1}_{j+1}", "Type A"))
                found_graphs = True
        
        if not found_graphs:
            extracted_graphs.append((pdf_path, i+1, "", "No Graph Found", "N/A"))
    
    return extracted_graphs

def create_graphs_table(pdf_files, output_folder):
    data = []
    for pdf in pdf_files:
        if pdf:  # Ensure the PDF file path is not empty
            graphs = extract_graphs_from_pdf(pdf, output_folder)
            data.extend(graphs)
        else:
            data.append(("", "", "", "", ""))  # Add empty row for empty PDF path
    
    df = pd.DataFrame(data, columns=["PDF Path", "Page Number", "Graph Path", "Graph Name", "Graph Type"])
    
    # Insert header row
    header_row = pd.DataFrame([["PDF Path", "Page Number", "Graph Path", "Graph Name", "Graph Type"]])
    df = pd.concat([header_row, df], ignore_index=True)
    
    return df

# Example usage
pdf_files = [
    "",
    "tests_on_inc_different_resolution/CGal_2_70_0.01_B_inc/CGal_2_70_0.01_B_inc_corner.pdf",
    "",
    "tests_on_inc_different_resolution/CGal_2_70_0.01_D_inc/CGal_2_70_0.01_D_inc_corner.pdf",
    "tests_on_inc_different_resolution/CGal_2_70_0.01_E_inc/CGal_2_70_0.01_E_inc_corner.pdf",
    "",
    "",
    "tests_on_inc_different_resolution/CGal_3_70_0.01_B_inc/CGal_3_70_0.01_B_inc_corner.pdf",
    "tests_on_inc_different_resolution/CGal_3_70_0.01_C_inc/CGal_3_70_0.01_C_inc_corner.pdf",
    "tests_on_inc_different_resolution/CGal_3_70_0.01_D_inc/CGal_3_70_0.01_D_inc_corner.pdf",
    "tests_on_inc_different_resolution/CGal_3_70_0.01_E_inc/CGal_3_70_0.01_E_inc_corner.pdf",
    "tests_on_inc_different_resolution/CGal_3_70_0.01_F_inc/CGal_3_70_0.01_F_inc_corner.pdf",
    "tests_on_inc_different_resolution/CGal_4_70_0.01_A_inc/CGal_4_70_0.01_A_inc_corner.pdf",
    "tests_on_inc_different_resolution/CGal_4_70_0.01_B_inc/CGal_4_70_0.01_B_inc_corner.pdf",
    "tests_on_inc_different_resolution/CGal_4_70_0.01_C_inc/CGal_4_70_0.01_C_inc_corner.pdf",
    "tests_on_inc_different_resolution/CGal_4_70_0.01_D_inc/CGal_4_70_0.01_D_inc_corner.pdf",
    "tests_on_inc_different_resolution/CGal_4_70_0.01_E_inc/CGal_4_70_0.01_E_inc_corner.pdf",
    "tests_on_inc_different_resolution/CGal_4_70_0.01_F_inc/CGal_4_70_0.01_F_inc_corner.pdf",
    "tests_on_inc_different_resolution/CGal_5_70_0.01_A_inc/CGal_5_70_0.01_A_inc_corner.pdf",
    "tests_on_inc_different_resolution/CGal_5_70_0.01_B_inc/CGal_5_70_0.01_B_inc_corner.pdf",
    "tests_on_inc_different_resolution/CGal_5_70_0.01_C_inc/CGal_5_70_0.01_C_inc_corner.pdf",
    "tests_on_inc_different_resolution/CGal_5_70_0.01_D_inc/CGal_5_70_0.01_D_inc_corner.pdf",
    "tests_on_inc_different_resolution/CGal_5_70_0.01_E_inc/CGal_5_70_0.01_E_inc_corner.pdf",
    "tests_on_inc_different_resolution/CGal_5_70_0.01_F_inc/CGal_5_70_0.01_F_inc_corner.pdf",
    "tests_on_inc_different_resolution/CGal_6_70_0.01_A_inc/CGal_6_70_0.01_A_inc_corner.pdf",
    "tests_on_inc_different_resolution/CGal_6_70_0.01_B_inc/CGal_6_70_0.01_B_inc_corner.pdf",
    "tests_on_inc_different_resolution/CGal_6_70_0.01_C_inc/CGal_6_70_0.01_C_inc_corner.pdf",
    "tests_on_inc_different_resolution/CGal_6_70_0.01_D_inc/CGal_6_70_0.01_D_inc_corner.pdf",
    "tests_on_inc_different_resolution/CGal_6_70_0.01_E_inc/CGal_6_70_0.01_E_inc_corner.pdf",
    "tests_on_inc_different_resolution/CGal_6_70_0.01_F_inc/CGal_6_70_0.01_F_inc_corner.pdf"
]  # Replace with actual PDF file paths

output_folder = "extracted_graphs"
df_graphs = create_graphs_table(pdf_files, output_folder)
print(df_graphs)
df_graphs.to_csv("graphs_table.csv", index=False, header=False)