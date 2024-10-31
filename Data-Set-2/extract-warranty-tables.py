import re
from bs4 import BeautifulSoup
import pandas as pd
import csv
import os
import html
import htmltabletomd
import json

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image as pil_image
import io
import base64

import openai
from pydantic import BaseModel, Field
from typing import Union, Literal, List, Dict


#--------------------------------------------------------------------------------------
# Step 1: Extract Warranty Tables
#--------------------------------------------------------------------------------------
"""
Function to read HTML from text files downloaded from SEC EDGAR and extract warranty table.
This code takes signifciant amount of time to process all the files in any given quarter. 
Depending on your computer speed, we are looking at anywhere from 6 to 10 hours for 10,000 files.

This function outputs three objects:

metadata (type: dictionary) contains the metadata of the file such as company name, CIK, etc.,
warranty table (type: HTML code) contains the HTML code of the warranty table
unit indicators is a list of dictionaries with all the mentions of units in the document. We have not used it in the data set but it is available. 

"""

def extract_warranty_table(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        
        # [Previous metadata extraction code remains unchanged...]
        metadata = {
            'ACCESSION NUMBER':
            re.search(r'ACCESSION NUMBER:\s*(\S+)', content),
            'CONFORMED SUBMISSION TYPE':
            re.search(r'CONFORMED SUBMISSION TYPE:\s*(\S+)', content),
            'CONFORMED PERIOD OF REPORT':
            re.search(r'CONFORMED PERIOD OF REPORT:\s*(\S+)', content),
            'FILED AS OF DATE':
            re.search(r'FILED AS OF DATE:\s*(\S+)', content),
            'COMPANY CONFORMED NAME':
            re.search(r'COMPANY CONFORMED NAME:\s*(.+)$',
                      content,
                      re.MULTILINE),
            'CENTRAL INDEX KEY':
            re.search(r'CENTRAL INDEX KEY:\s*(\S+)', content),
            'STANDARD INDUSTRIAL CLASSIFICATION':
            re.search(r'STANDARD INDUSTRIAL CLASSIFICATION:\s*(.+)$',
                      content,
                      re.MULTILINE)
        }
        metadata = {
            k: v.group(1).strip() if v else "Not found"
            for k, v in metadata.items()
        }

        document_match = re.search(r'<DOCUMENT>.*?</DOCUMENT>',
                                   content,
                                   re.DOTALL)
        if not document_match:
            print("No <DOCUMENT> tags found.")
            return metadata, None, []
        document_content = document_match.group()

        html_content = re.search(r'<HTML>.*?</HTML>', document_content,
                                 re.DOTALL | re.IGNORECASE)
        if html_content:
            soup = BeautifulSoup(html_content.group(), 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else "Not found"
            metadata['DOCUMENT TITLE'] = title
            
            # Find the Notes section
            notes_section = soup.find(
                lambda tag: tag.name in ['p', 'div', 'td'] and 
                tag.get_text() and 
                any(phrase.lower() in tag.get_text().lower() 
                    for phrase in ['Notes to Consolidated Financial Statements',
                                 'Notes to Condensed Consolidated Financial Statements',
                                 'Notes to Financial Statements'])
            )
            
            # Store unit indicators with their full sentences
            unit_indicators = []
            
            if notes_section:
                # Get all paragraphs and text elements after the Notes section
                elements = notes_section.find_all_next(['p', 'div', 'td', 'font', 'i'])
                
                # Define unit patterns
                unit_patterns = [
                    r'.*in thousands.*',
                    r'.*in millions.*',
                    r'.*\(000\).*omitted.*',
                    r'.*amounts.*thousands.*',
                    r'.*amounts.*millions.*',
                    r'.*\(000\'?s?\).*'
                ]
                
                # Search through each element
                for elem in elements:
                    text = elem.get_text().strip()
                    # Split into sentences (roughly)
                    sentences = re.split(r'[.!?](?=\s|$)', text)
                    
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                            
                        # Check each sentence against unit patterns
                        for pattern in unit_patterns:
                            if re.search(pattern, sentence, re.IGNORECASE):
                                # Clean up the sentence
                                clean_sentence = re.sub(r'\s+', ' ', sentence)
                                if clean_sentence not in [item['sentence'] for item in unit_indicators]:
                                    unit_indicators.append({
                                        'sentence': clean_sentence,
                                        'pattern_matched': pattern
                                    })
            
            warranty_terms = {
                # Base term -> all forms (including irregular plurals)
                'warranty accrual': ['warranty accrual', 'warranty accruals','warranties accrual'],
                'warranty expense': ['warranty expense', 'warranty expenses','warranties expenses'],
                'warranty liability': ['warranty liability', 'warranty liabilities', 'warranties liabiities'],
                'warranty claim': ['warranty claim', 'warranty claims','warranties claims'],
                'warranty reserve': ['warranty reserve', 'warranty reserves','warranties reserves'],
                'warranty cost': ['warranty cost', 'warranty costs','warranties costs']
            }
            # Flatten the list of all terms
            all_warranty_terms = [term for terms in warranty_terms.values() for term in terms]
            
            """If you want to search only in the notes section, uncomment the next line and indent the code following it."""
            #if notes_section:

            warranty_paragraphs = soup.find_all(
                lambda tag: tag.name in ['p', 'div'] and
                any(word in tag.get_text().lower() for word in all_warranty_terms) and
                'product' in tag.get_text().lower()
            )

            for paragraph in warranty_paragraphs:
                if any(phrase in paragraph.get_text().lower() for phrase in [
                    'changes in the product warranty',
                    'product warranty', 'warranties paid',
                    'warranty accrual', 'warranty accruals','warranties accrual',
                    'warranty expense', 'warranty expenses','warranties expenses',
                    'warranty liability', 'warranty liabilities', 'warranties liabiities',
                    'warranty claim', 'warranty claims','warranties claims',
                    'warranty reserve', 'warranty reserves','warranties reserves',
                    'warranty cost', 'warranty costs','warranties costs'
                ]):
                    potential_table = paragraph.find_next('table')
                    
                    if potential_table:
                        table_text = potential_table.get_text().lower()
                        
                        required_terms = [
                            'beginning',
                            'liability',
                            'accrual',
                            'period',
                            'claims',
                            'warranty',
                            'warranties'
                        ]
                        
                        exclusion_terms = [
                            'income statement',
                            'balance sheet',
                            'cash flow',
                            'revenue',
                            'inventory',
                            'loss',
                            'income',
                            'sales',
                            'assets',
                            'debt',
                            'dividend',
                            'lease',
                            'repair',
                            'stock', 'option', 'restricted', 'share', 'shares',
                            'interest', 'amortization',
                            'prepaid',
                            'research', 'administrative', 'marketing', 'advertising',
                            'benefit',
                            'dicount',
                            'returns',
                            'receivable',
                            'development',
                            'operating',
                            'pension',
                            'severance', 'abandonment', 'facilities', 'excess',
                            'goodwill', 'working', 'capital',
                            'employee', 'bonus', 'plan', 'salary', 'salaries',
                            'doubtful'
                        ]
                        
                        if (any(term in table_text for term in required_terms) and
                            not any(term in table_text for term in exclusion_terms) and
                            len(potential_table.find_all('tr')) > 2):
                            
                            dollar_pattern = re.compile(r'\$|\(\$?\d')
                            if dollar_pattern.search(table_text):
                                print("Found warranty table")
                                return metadata, potential_table, unit_indicators
                            
            print("No suitable warranty table found after examining all potential sections")
            return metadata, None, unit_indicators
        print("No HTML content found")
        return metadata, None, []

"""A helper function in case you want to validate the extracted warranty table. """
def validate_warranty_table(table):
    """Helper function to validate if a table is likely a warranty table"""
    table_text = table.get_text().lower()
    
    # Change these if necessary
    warranty_patterns = [
        r'beginning.*balance',
        r'accrual.*warranties',
        r'settlements made',
        r'ending.*balance',
        r'warranty.*(liability|reserve|provision)'
    ]
    
    pattern_matches = sum(1 for pattern in warranty_patterns 
                         if re.search(pattern, table_text))
    
    return pattern_matches >= 2


# Usage:
metadata, warranty_table, unit_indicators = extract_warranty_table('Your file path')


# Helper code to print the contents of unit_indicators

if unit_indicators:
    print("\nUnit Indicators found:")
    for idx, indicator in enumerate(unit_indicators, 1):
        print(f"\n{idx}. Sentence: {indicator['sentence']}")
        print(f"   Pattern matched: {indicator['pattern_matched']}")


# Helper function to convert unit indicators to a Pandas data frame

import pandas as pd

def convert_unit_indicators_to_df(unit_indicators):
    """
    Convert unit indicators list to a pandas DataFrame with additional parsed information.
    
    Args:
        unit_indicators (list): List of dictionaries containing unit indicator information
        
    Returns:
        pd.DataFrame: DataFrame containing parsed unit information
    """
    if not unit_indicators:
        return pd.DataFrame()
    
    # Create DataFrame from the list of dictionaries
    df = pd.DataFrame(unit_indicators)
    
    # Add column to identify the unit type
    df['unit_type'] = df['sentence'].apply(lambda x: 
        'thousands' if any(term in x.lower() for term in ['thousands', '(000', "000's"]) 
        else 'millions' if 'millions' in x.lower() 
        else 'unknown')
    
    # Add column for numeric multiplier
    df['multiplier'] = df['unit_type'].map({
        'thousands': 1000,
        'millions': 1000000,
        'unknown': None
    })
    
    # Reorder columns
    df = df[['sentence', 'unit_type', 'multiplier', 'pattern_matched']]
    
    return df

#--------------------------------------------------------------------------
# Step 2: Convert the warranty table into an image
#--------------------------------------------------------------------------

"""Define a function to capture screenshot of a single warranty table"""

def capture_table_screenshot(html_content, file_name, folder_path=".", encoding='utf-8'):
    # Ensure the folder path exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Create the full file path
    output_file = os.path.join(folder_path, f"{file_name}.png")

    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Set up the Chrome WebDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    html_content = html_content.prettify()

    try:
        # Handle the HTML content
        if isinstance(html_content, str):
            html_bytes = html_content.encode(encoding)
        elif isinstance(html_content, bytes):
            html_bytes = html_content
        else:
            raise ValueError("html_content must be either a string or bytes")
        
        # Create a complete HTML document with charset specified
        html_document = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="{encoding}">
            <title>Table Screenshot</title>
        </head>
        <body>
            {html_bytes.decode(encoding)}
        </body>
        </html>
        """.encode(encoding)
        
        html_base64 = base64.b64encode(html_document).decode('ascii')
        data_uri = f"data:text/html;charset={encoding};base64,{html_base64}"

        # Load the HTML content
        driver.get(data_uri)

        # Find the table element
        table = driver.find_element(By.TAG_NAME, 'table')

        # Capture the screenshot of the table
        screenshot = table.screenshot_as_png

        # Process the image using PIL
        image = pil_image.open(io.BytesIO(screenshot))

        # Save the image
        image.save(output_file)
        print(f"Screenshot saved as {output_file}")

    finally:
        driver.quit()


"""
The following code will use extract_warranty_table() and capture_table_screenshot() functions 
and apply them to all the text files in a folder. The output is stored in a subdirectory called "Images"
in the same folder. If this folder already has an Images subdirector, the function will save files inside it.
Otherwise it will create the subdirector.

The output files have .png extensions and the file name is the accession number. This allows us to match
the images with the SEC filings if required.

The function also returns a list of metadata ditctionaries and a second list of unit indicators.
"""

def screenshot_in_folder(folder_path):
    unit_indicators_list = []
    metadata_list = []
    image_folder = os.path.join(folder_path, "images")

        # Create 'images' subdirectory if it doesn't exist
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    for root, _, files in os.walk(folder_path):
        for idx, file in enumerate(files):
            if file.endswith(".txt"):  # Modify as needed for specific file types
                print(f"Processing file {idx + 1}: {file}")
                file_path = os.path.join(root, file)
                file_name_without_ext = os.path.splitext(file)[0]

                metadata, warranty_table, unit_indicators = extract_warranty_table(file_path)

                if warranty_table is not None:
                    try:
                        capture_table_screenshot(warranty_table, file_name_without_ext, image_folder, encoding='utf-8')
                        df = convert_unit_indicators_to_df(unit_indicators)
                        unit_indicators_list.append(df)
                        print(f"Unit indicators have been successfully written for '{file}'")
                        metadata_list.append(metadata)

                    except Exception as e:
                        print(f"Error taking screenshot for '{file}': {e}")
                else:
                    print(f"No warranty table found for file {idx + 1}: {file}.")
    return(metadata_list, unit_indicators_list)

# Usage example
metadata_list, unit_indicators_list =  screenshot_in_folder('Your Folder Path')

#----------------------------------------------------------------------------------------
# Step 3: Using OpenAI GPT 4o to extract warranty items from images
#----------------------------------------------------------------------------------------


# It is strongly recommended that you save the API key in the environment. However, you can also paste it below.
client = openai.OpenAI( api_key= 'Your API Key')


class WarrantyItem(BaseModel):
    warranty_item: str
    values: List[Union[int, None]]

class ReportingPeriod(BaseModel):
    period: str
    date: str
    year: int

"""The following text is what worked for us. You should experiment with it to find the most optimum prompts."""

class TableData(BaseModel):
    reporting_period: List[ReportingPeriod] = Field(description="Extracted reporting periods. Must be equal to num_col. For 'period' extract any text mentioning 'weeks ended' or 'months ended' or 'quarters ended'. Otherwise return NULL. Extract 'year' and 'date' from the column header. If there is no year in the column headers but the nottom most row of the table mentions a year, use it. Otherwise return NULL. If there is no date in the column headers but the nottom most row of the table mentions a date, use it. Otherwise return NULL.")
    warranty_items: List[WarrantyItem] = Field(description="The list of warranty items and their corresponding values. Do not combine warranty_items from different rows.")
    in_thousands: bool = Field(description="True if and only if the text explicitly states 'in thousands' or '(000) omitted'.")
    in_millions: bool = Field(description="True if an only if the text explicitly states 'in millions'")
    

# Updated DisclosureReport class with the custom Asset class
class DisclosureReport(BaseModel):
    assets: List[TableData]


# A function to encode the image before sending to OpenAI

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

""" In case you want to experiment with the prompt on individual images, use the following code"""

image_link = "Provide the full path to the image stored in the Images subdirectory"
my_image = encode_image(image_link)

# Extract the filename without the extension
accession_number = os.path.splitext(os.path.basename(image_link))[0]


input_messages = [
    {"role": "system", "content": "Output the result in JSON format."},
    {
        "role": "user",
        "content": [
            {"type": "text", 
             "text": "You are an image analyst. Your goal is to extract table in the image provided to you as a file."
             },
            {
                "type": "image_url",
                "image_url": 
                            {
                                "url": f"data:image/png;base64,{my_image}"
                            }
            },
        ],
    }
]

# gpt-4o-mini is cheap and fast and has vision capabilities. But we used gpt-4o as it gave better results.
response = client.beta.chat.completions.parse(
    response_format=DisclosureReport,
    model="gpt-4o",
    messages=input_messages
)

message = response.choices[0].message
# Print it out in readable format
obj = json.loads(message.content)
obj["accession number"] = accession_number


"""
Finally, here we convert all the images from "Images" subdirectory to a list of JSON objects returned from OpanAI.
This process was quick and took less than 6 seconds per image.
"""

def extract_table_in_folder(image_folder):
    json_list = []


    for root, _, files in os.walk(image_folder):
        for idx, file in enumerate(files):
            if file.endswith(".png"):  # Modify as needed for specific file types
                print(f"Processing file {idx + 1}: {file}")
                file_path = os.path.join(root, file)
                my_image = encode_image(file_path)
                # Extract the filename without the extension
                accession_number = os.path.splitext(os.path.basename(file_path))[0]
                input_messages = [
                    {"role": "system", "content": "Output the result in JSON format."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", 
                            "text": "You are a cool image analyst.  Your goal is to extract table in the image provided to you as a file."
                            },
                            {
                                "type": "image_url",
                                "image_url": 
                                            {
                                                "url": f"data:image/png;base64,{my_image}"
                                            }
                            },
                        ],
                    }
                ]

                # gpt-4o-mini is cheap and fast and has vision capabilities
                response = client.beta.chat.completions.parse(
                    response_format=DisclosureReport,
                    model="gpt-4o",
                    messages=input_messages
                )

                message = response.choices[0].message
                # Print it out in readable format   
                try:
                    obj = json.loads(message.content)
                    obj["accession number"] = accession_number
                except Exception as e:
                    print(f"skipping error: {e}")
                    continue


                json_list.append(obj)
    
    return(json_list)

#-------------------------------------------------------------------------------------
# Step 4: Convert the JSON objects to a table and save as Excel
#-------------------------------------------------------------------------------------

import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def validate_and_transform_warranty_data(json_list):
    """
    Transforms a list of JSON objects into a single pandas DataFrame,
    adjusting reporting periods and values according to length requirements.
    """
    all_dataframes = []
    validation_logs = []
    
    for json_index, data in enumerate(json_list, 1):
        try:
            for asset in data['assets']:
                rows = []
                reporting_periods = asset['reporting_period']
                
                for warranty_item in asset['warranty_items']:
                    values = warranty_item['values']
                    num_values = len(values)
                    num_periods = len(reporting_periods)
                    
                    # Case 1: reporting_period is shorter than values
                    if num_periods < num_values:
                        # Add NULL periods
                        null_periods = [
                            {
                                'period': None,
                                'date': None,
                                'year': None
                            }
                            for _ in range(num_values - num_periods)
                        ]
                        adjusted_periods = reporting_periods + null_periods
                        log_msg = (f"Added {num_values - num_periods} NULL reporting periods for "
                                 f"warranty item '{warranty_item['warranty_item']}' in "
                                 f"accession number {data.get('accession number', 'UNKNOWN')}")
                        validation_logs.append(log_msg)
                        logger.info(log_msg)
                    
                    # Case 2: reporting_period is longer than values
                    elif num_periods > num_values:
                        # Delete first few items from reporting_period
                        adjusted_periods = reporting_periods[num_periods - num_values:]
                        log_msg = (f"Removed first {num_periods - num_values} reporting periods for "
                                 f"warranty item '{warranty_item['warranty_item']}' in "
                                 f"accession number {data.get('accession number', 'UNKNOWN')}")
                        validation_logs.append(log_msg)
                        logger.info(log_msg)
                    
                    # Case 3: lengths match
                    else:
                        adjusted_periods = reporting_periods
                    
                    # Create rows with adjusted data
                    for idx, value in enumerate(values):
                        try:
                            rows.append({
                                'accession_number': data.get('accession number', 'UNKNOWN'),
                                'year': adjusted_periods[idx].get('year', None),
                                'period': adjusted_periods[idx].get('period', None),
                                'date': adjusted_periods[idx].get('date', None),
                                'warranty_item': warranty_item['warranty_item'],
                                'value': value,
                                'in_thousands': asset.get('in_thousands', None),
                                'in_millions': asset.get('in_millions', None)
                            })
                        except Exception as e:
                            logger.warning(f"Error processing row: {str(e)}")
                            continue
                
                if rows:
                    df = pd.DataFrame(rows)
                    all_dataframes.append(df)
                
        except Exception as e:
            logger.warning(f"Error processing JSON object {json_index}: {str(e)}")
            continue
    
    if not all_dataframes:
        logger.warning("No valid data was processed. Returning empty DataFrame.")
        return pd.DataFrame(columns=[
            'accession_number', 'year', 'period', 'date', 
            'warranty_item', 'value', 'in_thousands', 'in_millions'
        ])
    
    # Concatenate all DataFrames
    final_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Reorder columns
    column_order = [
        'accession_number',
        'year',
        'period',
        'date',
        'warranty_item',
        'value',
        'in_thousands',
        'in_millions'
    ]
    final_df = final_df[column_order]
    
    # Print summary of validation logs
    if validation_logs:
        print("\nValidation Adjustments Summary:")
        for log in validation_logs:
            print(log)
    
    return final_df

# Usage example

validate_and_transform_warranty_data("The List of JSON objects returned from OpenAI").to_excel('File path where you want to save the output with xlsx extension')

