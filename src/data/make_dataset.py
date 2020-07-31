# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import os
import urllib
from PIL import Image

def download_data(data):
  '''Downloads car images'''

  dataset = dict()
  dataset["image_name"] = list()
  dataset["image_width"] = list()
  dataset["image_height"] = list()
  dataset["top_x"] = list()
  dataset["top_y"] = list()
  dataset["bottom_x"] = list()
  dataset["bottom_y"] = list()

  #Downloading images
  counter = 0
  print('\nDownloading car images')
  for index, row in data.iterrows():
      file_name = "data/raw/"+"Indian_Number_Plates/{}.jpg".format(counter)
      
      dataset["image_name"].append("{}".format(counter))
      
      data = row["annotation"]
      dataset["image_width"].append(data[0]["imageWidth"])
      dataset["image_height"].append(data[0]["imageHeight"])
      dataset["top_x"].append(data[0]["points"][0]["x"])
      dataset["top_y"].append(data[0]["points"][0]["y"])
      dataset["bottom_x"].append(data[0]["points"][1]["x"])
      dataset["bottom_y"].append(data[0]["points"][1]["y"])
      
      counter += 1
      
      if os.path.exists(file_name):
        continue

      img = urllib.request.urlopen(row["content"])
      img = Image.open(img)
      img = img.convert('RGB')
      img.save(file_name, "JPEG")
  print("Downloaded {} car images.\n".format(counter))
  return pd.DataFrame(dataset)


@click.command()
@click.option("--input-path", "-i", "input_path", default="data/external/",
    help="Path to json file to be read.")
@click.option("--output-path", "-o", default="data/raw/",
    help="Path to csv file to store the result.")

def main(input_path, output_path):
    """ Runs data processing scripts to read external data (../external) and convert it into
        csv file (../raw) to be further processed.
    """
    logger = logging.getLogger(__name__)
    logger.info('making raw data')
    
    plates_json_df = pd.read_json(input_path+"Indian_Number_plates.json", lines=True)
    
    #Downloading car images
    os.makedirs(output_path+"Indian_Number_Plates", exist_ok = True)
    df = download_data(plates_json_df)
	
	  # Serialize the dataframe
    df.to_csv(output_path+"indian_license_plates.csv", index=False)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
 
    main()