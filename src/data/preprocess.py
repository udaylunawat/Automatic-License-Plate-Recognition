import click
import pandas as pd

@click.command()
@click.option("--input-path", "-i", default = "data/raw/", required=True,
    help="Path to csv file to be processed.",
)
@click.option("--output-path", "-o", default="data/preprocessed/",
    help="Path to csv file to store the result.")

def main(input_path, output_path):
    """ Runs data processing scripts to read raw data (../raw) and convert it into
        processed csv file (../processed) to be used for further analysis.
    """
    print("Preprocessing indian_license_plate.csv")
    df = pd.read_csv(input_path+"indian_license_plates.csv", dtype={'image_name':str})
    df["image_name"] = df["image_name"] + ".jpg"
    df.to_csv(output_path+"processed.csv", index=False)
    print("Preprocessed and saved as processed.csv")
    
if __name__ == '__main__':
	main()
	