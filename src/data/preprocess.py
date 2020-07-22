import click
import pandas as pd

@click.command()
@click.option("--in-file", "-i", default = "data/raw/", required=True,
    help="Path to csv file to be processed.",
)
@click.option("--out-file", "-o", default="data/processed/processed.csv",
    help="Path to csv file to store the result.")

def main(in_file, out_file):
    '''
    '''
    df = pd.read_csv(in_file+"/indian_license_plates.csv")
    df["image_name"] = df["image_name"] + ".jpeg"
    df.drop(["image_width", "image_height"], axis=1, inplace=True)
    df.to_csv(out_file+"/processed.csv", index=False)
    
if __name__ == '__main__':
	print("Preprocessing indian_license_plate.csv")
	main()
	print("Preprocessed and saved as preprocessed.csv")