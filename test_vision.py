from image_description import describe_image_with_gemini
desc, meta = describe_image_with_gemini(
    "pdfs/27/images/img_010_01.jpeg"
)

print(desc)
