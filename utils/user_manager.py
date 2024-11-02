import datetime
import os
import csv

def generate_unique_id():
    current_time = datetime.datetime.now()
    unique_id = current_time.strftime("%Y%m%d%H%M%S")
    return unique_id

def save_user_to_csv(name):
    user_id = generate_unique_id()
    date_created = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_file = 'user_data.csv'
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['ID', 'Name', 'Date Created'])
        writer.writerow([user_id, name, date_created])
    return user_id

def create_user_directory(user_id):
    directory = f"dataset/{user_id}"
    try:
        os.makedirs(directory)
        print(f"Directory {directory} created successfully")
    except Exception as e:
        print(f"Error creating directory {directory}: {e}")
        
def get_username_from_csv(user_id):
    with open('user_data.csv', mode="r") as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if row["ID"] == user_id:
                return row['Name']
    return None
