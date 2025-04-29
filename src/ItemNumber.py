from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import os
import requests
from PIL import Image
from io import BytesIO

# Set up ChromeDriver (Update path if needed)
chrome_driver_path = r"/chromedriver-win64/chromedriver.exe"
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service)

# URL to scrape
url = "https://berkshireproducts.com/species3_as.php?species%5B%5D=allspec&w_over=&w_under=&l_over=&l_under="
driver.get(url)
time.sleep(10)  # Allow more time for the page to load

# Scroll to load table rows
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(3)

# Extract item numbers from table rows
item_numbers = []
table_rows = driver.find_elements(By.XPATH, "//table[@id='table']//tr")
print(f"Found {len(table_rows)} table rows.")

for i, row in enumerate(table_rows[:20]):
    cells = row.find_elements(By.TAG_NAME, "td")
    if len(cells) > 1:
        item_number = cells[1].text.strip()
        if item_number:
            item_numbers.append(item_number)

print(f"Collected item numbers: {item_numbers}")

# Download images for the first 20 rows
for item_number in item_numbers:
    try:
        first_two_digits = item_number[:2]
        image_url = f"https://berkshireproducts.com/images/inventory/{first_two_digits}/{item_number}-lcr.jpg"
        response = requests.get(image_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            # Convert RGBA to RGB if needed
            if image.mode == "RGBA":
                image = image.convert("RGB")
            # Save image
            os.makedirs(".venv/Scripts/wood_species_images", exist_ok=True)
            image_path = os.path.join(".venv/Scripts/wood_species_images", f"{item_number}.jpg")
            image.save(image_path)
            print(f"Saved: {image_path}")
        else:
            print(f"Image not found: {image_url}")
    except Exception as e:
        print(f"Error downloading image for item #{item_number}: {e}")

# Close the browser
driver.quit()
