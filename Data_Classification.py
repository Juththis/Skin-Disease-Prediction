import pandas as pd
import shutil
import os

# Load the dataset
data = pd.read_csv("HAM10000_metadata.csv")

data.drop(columns=["lesion_id","dx_type","age","sex","localization"], inplace=True  )

print(data.groupby("dx")["image_id"].apply(list))

grouped = data.groupby("dx")["image_id"].apply(list)
os.makedirs("akiec", exist_ok=True)
os.makedirs("bcc", exist_ok=True)
os.makedirs("bkl", exist_ok=True)
os.makedirs("df", exist_ok=True)
os.makedirs("mel", exist_ok=True)
os.makedirs("nv", exist_ok=True)
os.makedirs("vasc", exist_ok=True)

for item in grouped["akiec"]:
    source = item + ".jpg"
    destination = os.path.join("akiec", item + ".jpg")
    shutil.copy(source, destination)

for item in grouped["bcc"]:
    source = item + ".jpg"
    destination = os.path.join("bcc", item + ".jpg")
    shutil.copy(source, destination)

for item in grouped["bkl"]:
    source = item + ".jpg"
    destination = os.path.join("bkl", item + ".jpg")
    shutil.copy(source, destination)

for item in grouped["df"]:
    source = item + ".jpg"
    destination = os.path.join("df", item + ".jpg")
    shutil.copy(source, destination)

for item in grouped["mel"]:
    source = item + ".jpg"
    destination = os.path.join("mel", item + ".jpg")
    shutil.copy(source, destination)

for item in grouped["nv"]:
    source = item + ".jpg"
    destination = os.path.join("nv", item + ".jpg")
    shutil.copy(source, destination)

for item in grouped["vasc"]:
    source = item + ".jpg"
    destination = os.path.join("vasc", item + ".jpg")
    shutil.copy(source, destination)


