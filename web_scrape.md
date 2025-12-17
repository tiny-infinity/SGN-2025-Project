### Web Basics

Websites are made up of three main ingredients - HTML (HyperTextMarkupLanguage), CSS (Cascading Style Sheets), Javascript.

Static websites are those where eveything is a HTML element. 

- HTML is made up of "elements". Everything is within tags (e.g'''<title> SGN <title>''')
- CSS : Defines visual style of the webpage - fonts, layout, colours etc. Comes in three types - inline, global 
- JavaScript allows interactivity in pages - animations, buttons etc. 

To steal data, we need to know fundamentals of all the above. 

Note : We only have access to the frontend, even when webscraping. 

The DOM (Document Object Model) is a way of rperesenting the structure of a webpage. 

#### Tools and Libraries

1. BeautifulSoup
Python package used for parsing HTML data. Removes all tags and gives text. Cannot download webpages on its own. Have to request for the HTML file first. Offers multiple ways to navigate and search the parse tree. Can find elements based on their attributes.

2. Requests
Python package that can request HTML files using URLs.

3. Scrapy
Used for more complex web scraping projects, non-static websites. 

4. Selenium 
For dynamic webpages, ones with lazy loading (not loading whole page at once). For completely Java-based.






