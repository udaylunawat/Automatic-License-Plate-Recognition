import streamlit as st
import pytesseract
# import easyocr

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=True)
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
                custom_config = r'--oem {} --psm {}'.format(oem,psm)
                text_output = pytesseract.image_to_string(crop_image, config=custom_config)
                st.warning(custom_config+':'+text_output)
                progress_bar.progress(counter/(4*14))

            except:
                continue



def easy_OCR(crop):
    reader = easyocr.Reader(['en'])
    ocr_output = reader.readtext(np.array(crop))
    plate_text = ''
    for text in ocr_output:
        plate_text += text[1]
    return plate_text

def OCR(crop_image):
    # psm 6 - single line license
    try:
        custom_config = r'--oem 3 --psm 9'
        text_output = pytesseract.image_to_string(crop_image, config=custom_config)
        print(custom_config,':',text_output)
    except:
        pass
    return text_output