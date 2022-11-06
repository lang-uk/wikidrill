# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class PwnFriendsItem(scrapy.Item):
    id_from = scrapy.Field()
    id_to = scrapy.Field()
    rel = scrapy.Field()
