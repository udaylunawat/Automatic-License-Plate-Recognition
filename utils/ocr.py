import streamlit as st
import pytesseract
import numpy as np
try:
    import easyocr
    reader = easyocr.Reader(['en'])
except:
    pass

def try_all_OCR(crop_image):
    st.write(" **OEM**: Engine mode \
             \n**PSM**: Page Segmentation Mode \
             \n Click [here](https://nanonets.com/blog/ocr-with-tesseract/) to know more!")

    progress_bar = st.progress(0)
    counter = 0
    for oem in range(0,4):
        for psm in range(0,14):
            counter += 1
            try:
                custom_config = r'-l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --oem {} --psm {}'.format(oem,psm)
                text_output = pytesseract.image_to_string(crop_image, config=custom_config)
                st.warning("oem: {} psm: {}: {}".format(oem, psm, text_output))
                progress_bar.progress(counter/(4*14))
            except:
                continue

def easy_OCR(crop_image):
    ocr_output = reader.readtext(np.array(crop_image))
    text_output = ''
    for text in ocr_output:
        text_output += text[1]
    return text_output

def OCR(crop_image):
    text_output = ''
    try:
        custom_config = r'-l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --oem 1 --psm 6'
        text_output = pytesseract.image_to_string(crop_image, config=custom_config)
        print(custom_config,':',text_output)
    except:
        pass
    return text_output


