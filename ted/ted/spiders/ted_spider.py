# -*- coding: utf-8 -*-
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from lxml import html


class Transcript(scrapy.Item):
    link = scrapy.Field()
    text = scrapy.Field()
    

class TedSpiderSpider(CrawlSpider):
    name = 'ted_spider'
    allowed_domains = ['www.ted.com']
    start_urls = ['https://www.ted.com/talks?language=sr&page=1']
    
    BASE_URL = "https://www.ted.com"
        
    rules = (Rule(LinkExtractor(allow=(), restrict_xpaths=('//a[@rel="next"]',)), callback="parse_page", follow= True),)


    def parse_start_url(self, response):
        site = html.fromstring(response.body_as_unicode())
        talks = site.xpath('//a[@data-ga-context="talks"]')

        for talk in talks:
            suffix = talk.xpath('./@href')[0].split("?")[0]
            absolute_url = self.BASE_URL + suffix + "/transcript?language=sr"
            yield scrapy.Request(absolute_url, callback=self.parse_transcript)
            
    def parse_page(self, response):
        site = html.fromstring(response.body_as_unicode())
        talks = site.xpath('//a[@data-ga-context="talks"]')

        for talk in talks:          
            suffix = talk.xpath('./@href')[0].split("?")[0]
            absolute_url = self.BASE_URL + suffix + "/transcript?language=sr"
            yield scrapy.Request(absolute_url, callback=self.parse_transcript)

    def parse_transcript(self, response):
        item = Transcript()
        item['link'] = response.url
        
        site = html.fromstring(response.body_as_unicode())
        text = site.xpath('//div[@class="Grid__cell flx-s:1 p-r:4"]/p/text()')
        item['text'] = text
        
        return item 
        
        