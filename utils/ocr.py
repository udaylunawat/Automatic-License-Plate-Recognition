import streamlit as st
import pytesseract
import re
import easyocr
reader = easyocr.Reader(['en'])
import numpy as np

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
    plate_text = ''
    for text in ocr_output:
        plate_text += text[1]
    return plate_text

def OCR(crop_image):
    # psm 6 - single line license
    try:
        custom_config = r'-l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --oem 1 --psm 6'
        text_output = pytesseract.image_to_string(crop_image, config=custom_config)
        print(custom_config,':',text_output)
    except:
        pass
    return text_output


