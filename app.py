import importlib.util
import os
import requests
import nltk
import openai
from bs4 import BeautifulSoup
from urllib.parse import urlencode
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
from spinners import Spinners
import textwrap
import datetime
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import random
import time
from termcolor import colored
from colorama import init, Fore, Back, Style

# Initialize colorama
init()

#----------------------- INITIAL SETUP -----------------------#
required_packages = [
    "psutil",
    "GPUtil",
    "recursive",
    "termcolor",
    "prettytable",
    "textwrap",
    "tqdm",
    "spinners"
]

for package in required_packages:
    spec = importlib.util.find_spec(package)
    if spec is None:
        print(f"Installing {package}...")
        try:
            import pip
            pip.main(["install", package])
        except:
            print(f"Failed to install {package}. Please install it manually.")

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Initialize the sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Initialize the WebDriver for Chrome
driver = webdriver.Chrome(ChromeDriverManager().install())

openai.api_key = 'sk-x'
api_key = os.environ.get('openai_api_key') 

#----------------------- GENERATING SEARCH QUERIES -----------------------#

# Generate search queries based on the given keywords
def generate_search_queries(keywords):
    queries = [f"{keyword} blog post" for keyword in keywords]
    return queries


#----------------------- GETTING SEARCH URL -----------------------#

# Get the Google search URL for the given query
def get_search_url(query):
    query = urlencode({'q': query})
    url = f"https://www.google.com/search?{query}"
    return url


#----------------------- GETTING RESULT LINKS -----------------------#

# Get the URLs of search result links from the Google search page
def get_result_links(search_url, driver):
    driver.get(search_url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    result_containers = soup.find_all('div', class_='g')
    for container in result_containers:
        link_tag = container.find('a')
        if link_tag:
            yield link_tag.get('href')
        time.sleep(random.randint(1, 2))


#----------------------- EXTRACTING CONTENT -----------------------#

# Extract the text content from a webpage URL
def extract_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup.get_text()
    except:
        print(f"{Fore.RED}An error occurred while extracting content from the URL: {url}{Style.RESET_ALL}")
        return None


#----------------------- EXTRACTING KEYWORDS -----------------------#

# Extract keywords from the content using sentiment analysis
def extract_keywords(content):
    if content is not None:
        words = re.findall(r'\w+', content.lower())
        scored_words = sentiment_analyzer.polarity_scores(' '.join(words))
        keywords = [word for word, score in scored_words.items() if score > 0.2]
        return keywords
    else:
        return []


#----------------------- GENERATING OPENAI SUMMARY -----------------------#

# Generate a summary of the given text using OpenAI's API
def get_openai_summary(text, api_key, max_tokens=3000, chunk_size=3000):
    
    chunks = textwrap.wrap(text, chunk_size)
    summary_chunks = []
    with tqdm(total=len(chunks), desc=Fore.YELLOW + "Generating summary",
              bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.YELLOW, Style.RESET_ALL)) as pbar:
        for chunk in chunks:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=chunk,
                max_tokens=max_tokens,
                temperature=0.5,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=None
            )
            summary_chunks.extend(response.choices[0].text.strip().split("\n"))
            pbar.update(1)
    return summary_chunks


#----------------------- GENERATING BLOG POST -----------------------#

# Generate a blog post based on the given keywords and search results
def generate_blog_post(keywords, num_results, api_key, max_tokens, driver):
    blog_post = ""

    # Generate search queries for the keywords
    search_queries = generate_search_queries(keywords)

    # Store trends in a dictionary
    trends = {}

    for keyword, query in zip(keywords, search_queries):
        blog_post += f"# {keyword}\n\n"

        # Perform Google searches for each search query
        search_url = get_search_url(query)
        search_results = list(get_result_links(search_url, driver))[:num_results]
        for url in search_results:
            content = extract_content(url)
            if content is not None:
                blog_post += f"{content}\n\n"

                # Analyze trends over time
                trends[keyword] = trends.get(keyword, 0) + content.count(keyword)

    # Parse the scraped content with BeautifulSoup
    soup = BeautifulSoup(blog_post, 'html.parser')
    cleaned_blog_post = soup.get_text()

    # Generate OpenAI summary for the cleaned blog post
    summary_chunks = get_openai_summary(cleaned_blog_post, api_key, max_tokens)
    summary = "\n".join(summary_chunks)

    return summary, trends


#----------------------- SAVING CONTENT TO FILES -----------------------#

# Save the content to a Markdown file
def save_to_markdown(content, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)


# Save the trends to a text file
def save_trends_to_file(trends, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for keyword, count in trends.items():
            file.write(f"{keyword}: {count}\n")


#----------------------- USER ONBOARDING -----------------------#

# Perform the user onboarding process
def user_onboarding(driver):
    keywords = input("Enter the keywords for the blog post (separated by commas): ").split(",")
    num_results = 5
    api_key = input("Enter your OpenAI API key: ")
    max_tokens = 100
    print("Generating blog post...\n")
    summary, trends = generate_blog_post(keywords, num_results, api_key, max_tokens, driver)
    save_to_markdown(summary, "blog_post.md")
    print("Blog post saved to 'blog_post.md' file.")
    save_trends_to_file(trends, "keyword_trends.txt")
    print("Keyword trends saved to 'keyword_trends.txt' file.")


if __name__ == "__main__":
    # Print system information and current timestamp
    print(f"{Fore.YELLOW}Author: Rajan Selvan")
    print(f"System Information: Python {os.sys.version}")
    print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")

    # Initialize the Chrome driver
    driver = webdriver.Chrome(ChromeDriverManager().install())

    # Perform user onboarding
    user_onboarding(driver)

    # Quit the Chrome driver
    driver.quit()
