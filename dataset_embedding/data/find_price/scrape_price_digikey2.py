import pandas as pd
import time
import random
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException,WebDriverException
from selenium_stealth import stealth
import os

class text_to_be_different_from:
    def __init__(self, locator, old_text):
        self.locator = locator
        self.old_text = old_text

    def __call__(self, driver):
        try:
            current_text = driver.find_element(*self.locator).text
            return current_text != self.old_text
        except (NoSuchElementException, StaleElementReferenceException):
            return True # 如果元素暫時消失再出現，也視為已變化
        except Exception:
            return False

class DigiKeyScraper:
    """
    一個專門用於爬取 Digi-Key 網站電容資訊的爬蟲類別。
    """
    # 將所有定位器 (selectors) 集中管理，方便未來維護
    SELECTORS = {
        "cookie_accept_button": (By.CSS_SELECTOR, 'button[id="onetrust-reject-all-handler"]'),
        "filter_template": (By.XPATH, '//span[text()="{filter_name}"]'),
        "apply_filters_button": (By.XPATH, "//button[normalize-space()='全部应用']"),
        #"product_table": (By.ID, "product-table"),
        "product_rows": (By.CSS_SELECTOR, "div[data-testid='sb-content-container'] table > tbody > tr"),
        "price_data_cell": (By.CSS_SELECTOR, 'td[data-testid*="price-and-qty"]'),  # 使用 * 通配符增加穩健性
        "price_data_span": (By.CSS_SELECTOR, 'span[data-qty]'),
        "next_page_button": (By.CSS_SELECTOR, 'button[data-testid="btn-next-page"]'),
        "manufacturer_scroll_container": (
        By.CSS_SELECTOR, 'div[data-testid="filter-box-group--1"] > div[style*="overflow: auto"]'),
        "loading_cover": (By.CSS_SELECTOR, "div[data-testid='loadingCover']"),
        "first_product_row": (By.CSS_SELECTOR, "div[data-testid='sb-content-container'] table > tbody > tr:first-child"),
        "pagination_indicator": (By.CSS_SELECTOR, "div[data-testid='per-page-selector'] > div[role='button']"),
    }

    def __init__(self, driver_path: str):
        """
        初始化爬蟲。

        Args:
            driver_path (str): chromedriver.exe 的絕對路徑。
        """
        self.driver_path = driver_path
        self.driver = None
        self.wait = None

    def find_and_click_in_scrollable_list(self, container_locator, target_locator):
        """
        在一個可滾動的容器內，不斷向下滾動，直到找到目標元素並點擊它。

        Args:
            container_locator: 可滾動容器的定位器 (一個 tuple)。
            target_locator: 內部目標元素的定位器 (一個 tuple)。
        """
        print(f"正在尋找滾動容器...")
        # 首先定位到那個可以滾動的 div 容器
        scroll_container = self.wait.until(EC.presence_of_element_located(container_locator))

        last_scroll_height = -1  # 用於判斷是否已滾動到底部

        # 設置一個循環來嘗試滾動和尋找
        for _ in range(10):  # 最多嘗試滾動 10 次，防止無限循環
            try:
                # 1. 先嘗試直接尋找元素，如果它一開始就在視野內
                time.sleep(0.5)
                target_element = self.driver.find_element(*target_locator)
                print(f"已找到目標元素 '{target_locator[1]}'")
                target_element.click()
                return  # 找到並點擊後，函式結束
            except NoSuchElementException:
                # 2. 如果找不到，則滾動容器
                print("未找到，正在向下滾動...")
                # 使用 JavaScript 來執行滾動操作，更可靠
                self.driver.execute_script("arguments[0].scrollTop += 50;", scroll_container)
                time.sleep(4)  # 等待滾動後的新內容加載

                # 檢查是否已滾動到底部
                current_scroll_height = self.driver.execute_script("return arguments[0].scrollTop;", scroll_container)
                if current_scroll_height == last_scroll_height:
                    print("已滾動到底部，但未找到目標元素。")
                    break
                last_scroll_height = current_scroll_height

        # 如果循環結束還沒找到，就拋出異常
        raise TimeoutException(f"在滾動容器內最終也未能找到目標元素: {target_locator[1]}")


    def _initialize_driver(self):
        """初始化 Selenium WebDriver 和 Stealth。"""
        print("正在初始化 WebDriver...")
        options = webdriver.ChromeOptions()
        options.add_argument("start-maximized")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)

        service = ChromeService(executable_path=self.driver_path)
        self.driver = webdriver.Chrome(service=service, options=options)
        self.wait = WebDriverWait(self.driver, 120)

        stealth(self.driver, languages=["en-US", "en"], vendor="Google Inc.", platform="Win32")
        print("WebDriver 初始化完成。")

    def _handle_cookies(self):
        """處理 Cookie 同意彈窗。"""
        try:
            print("正在檢查 Cookie 同意按鈕...")
            #cookie_button = self.wait.until(EC.element_to_be_clickable(self.SELECTORS["cookie_accept_button"]))
            cookie_reject_button = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[id="onetrust-reject-all-handler"]'))
            )
            cookie_reject_button.click()
            print("已點擊 Cookie 同意按鈕。")
            time.sleep(1)
        except TimeoutException:
            print("未找到 Cookie 同意按鈕，繼續執行。")

    def apply_filters(self, filters: dict):
        """
        應用篩選條件。

        Args:
            filters (dict): 一個包含篩選條件的字典，例如 {'制造商': 'KEMET'}
        """
        try:
            for filter_type, filter_name in filters.items():
                print(f"正在應用篩選器 '{filter_type}': '{filter_name}'...")

                # 使用模板動態生成定位器
                locator = (self.SELECTORS["filter_template"][0],
                           self.SELECTORS["filter_template"][1].format(filter_name=filter_name))
                self.find_and_click_in_scrollable_list(
                    container_locator=self.SELECTORS["manufacturer_scroll_container"],
                    target_locator=locator # 動態生成定位器
                )
                #filter_element = self.wait.until(EC.element_to_be_clickable(('xpath','//span[text()="KEMET"]')))
                filter_element = self.wait.until(EC.element_to_be_clickable(locator))
                #filter_element.click()
                print(f"已成功選擇 '{filter_name}'。")
                time.sleep(1)  # 模擬人類點擊間隔

            print("正在點擊 '全部应用' 按鈕...")
            apply_button = self.wait.until(EC.element_to_be_clickable(self.SELECTORS["apply_filters_button"]))
            apply_button.click()
            print("已點擊 '全部应用' 按鈕。")

            # print("正在等待篩選結果刷新...")
            # self.wait.until(EC.presence_of_element_located(self.SELECTORS["product_rows"]))
            # print("篩選結果已成功加載。")
            time.sleep(20) #等待加载完成

        except TimeoutException as e:
            print(f"尋找或點擊篩選器時超時: {e}")
            raise e  # 拋出異常，讓主流程知道篩選失敗

    def _parse_page(self, page_source: str) -> list:
        """解析單一頁面的 HTML，提取產品數據。"""
        soup = BeautifulSoup(page_source, 'html.parser')
        products_on_page = []

        product_rows = soup.select(self.SELECTORS["product_rows"][1])  # 使用 CSS Selector

        for row in product_rows:
            target_td = row.select_one('td[data-testid="draggable-cell--99"]')
            # 如果在該行中找到了這個特殊的 td
            if target_td:
                # 5. 在這個 td 內部，直接尋找那個帶有 data-qty 屬性的 span 標籤
                # 這是一個很穩健的定位方式，因為我們知道目標 span 有這個屬性
                target_span = target_td.find('span', attrs={'data-qty': True})
                # 如果找到了目標 span
                if target_span:
                    try:
                        # 6. 從 span 的屬性中提取所有我們需要的數據
                        # .get(key, default_value) 是一種安全的取值方式，如果屬性不存在則返回預設值
                        product_data = {
                            'mfg_number': target_span.get('data-mfg-number', 'N/A'),
                            'quantity': target_span.get('data-qty', 'N/A'),
                            'price': target_span.get('data-price', 'N/A'),
                            'mfg_name': target_span.get('data-mfg-name', 'N/A'),
                            'description': target_span.get('data-desc', 'N/A')
                        }
                        products_on_page.append(product_data)
                    except Exception as e:
                        print(f"在提取 span 屬性時發生錯誤: {e}")
                else:
                    # 在某些行（例如分頁提示行）可能沒有這個 span，這很正常
                    pass
            else:
                # 在某些行（例如表格中的分隔行）可能沒有這個 td，這也很正常
                pass




        return products_on_page

    # def scrape_category(self, start_url: str, filters: dict, max_pages: int = 500, output_csv: str = "digikey_data.csv",
    #                     resume: bool = True):
    #     """
    #     爬取一個指定類別的完整流程，增加了重試和斷點續爬功能。
    #
    #     Args:
    #         start_url (str): 起始 URL。
    #         filters (dict): 篩選條件。
    #         max_pages (int): 最大爬取頁數。
    #         output_csv (str): 輸出 CSV 檔案的路徑。
    #         resume (bool): 是否嘗試從現有 CSV 檔案斷點續爬。
    #     """
    #     all_products = []
    #     start_page = 1
    #
    #     # --- 斷點續爬邏輯 ---
    #     if resume and os.path.exists(output_csv):
    #         try:
    #             # 簡單地透過計算已爬取的頁數來繼續 (假設每頁25條)
    #             # 更穩健的做法是額外保存一個進度檔案
    #             df_existing = pd.read_csv(output_csv)
    #             num_scraped = len(df_existing)
    #             start_page = (num_scraped // 25) + 1  # 假設每頁 25 條數據
    #             all_products = df_existing.to_dict('records')
    #             print(f"發現已存在的檔案 '{output_csv}'，其中包含 {num_scraped} 條數據。")
    #             print(f"將從第 {start_page} 頁繼續爬取...")
    #         except Exception as e:
    #             print(f"讀取現有檔案時出錯: {e}，將從頭開始。")
    #
    #     self._initialize_driver()
    #
    #     try:
    #         if start_page == 1:
    #             # 只有在從頭開始時才需要訪問起始頁和應用篩選器
    #             self.driver.get(start_url)
    #             self._handle_cookies()
    #             if filters:
    #                 self.apply_filters(filters)
    #         else:
    #             # 如果是續爬，需要先跳轉到對應的頁面
    #             # 這裡假設 URL 結構簡單，實際可能需要多次點擊
    #             # 為了簡化，我們這裡還是從第一頁開始點擊，直到目標頁面
    #             self.driver.get(start_url)
    #             self._handle_cookies()
    #             if filters: self.apply_filters(filters)
    #
    #             print(f"正在跳轉至第 {start_page} 頁...")
    #             for _ in range(1, start_page):
    #                 next_page_button = self.wait.until(EC.element_to_be_clickable(self.SELECTORS["next_page_button"]))
    #                 self.driver.execute_script("arguments[0].click();", next_page_button)
    #                 self.wait.until(EC.invisibility_of_element_located(self.SELECTORS["loading_cover"]))
    #             print("已成功跳轉。")
    #
    #         for page_num in range(start_page, max_pages + 1):
    #             # --- 重試機制 ---
    #             current_page_success = False
    #             for attempt in range(1, 4):  # 最多重試 3 次
    #                 try:
    #                     print(f"\n--- 正在處理第 {page_num} 頁 (嘗試第 {attempt} 次) ---")
    #
    #
    #
    #                     # 等待並解析頁面
    #                     self.wait.until(EC.presence_of_element_located(self.SELECTORS["product_rows"]))
    #                     time.sleep(random.uniform(1, 3))  # 增加隨機延遲
    #
    #                     page_source = self.driver.page_source
    #
    #                     # 檢查是否為伺服器錯誤頁面
    #                     if "抱歉，看來我們遇到了問題" in page_source or "Internal Server Error" in page_source:
    #                         raise WebDriverException("伺服器返回了 500 錯誤頁面。")
    #
    #                     products_on_page = self._parse_page(page_source)
    #                     all_products.extend(products_on_page)
    #                     print(f"在本頁找到 {len(products_on_page)} 條產品資訊。")
    #
    #                     # --- 定期保存進度 ---
    #                     if page_num % 5 == 0:  # 每 5 頁保存一次
    #                         df_temp = pd.DataFrame(all_products)
    #                         df_temp.to_csv(output_csv, index=False, encoding='utf-8-sig')
    #                         print(f"進度已保存到第 {page_num} 頁。")
    #
    #                     current_page_success = True
    #                     break  # 當前頁面成功，跳出重試迴圈
    #
    #                 except WebDriverException as e:
    #                     print(f"處理第 {page_num} 頁時發生錯誤: {e}")
    #                     if attempt < 3:
    #                         wait_time = 5 * attempt  # 等待時間逐漸增加
    #                         print(f"將在 {wait_time} 秒後重試...")
    #                         time.sleep(wait_time)
    #                         self.driver.refresh()  # 嘗試刷新頁面
    #                     else:
    #                         print("重試次數已達上限，放棄當前頁面。")
    #
    #             if not current_page_success:
    #                 # 如果重試 3 次後依然失敗，可以選擇跳過這一頁或終止程式
    #                 print(f"跳過第 {page_num} 頁。")
    #
    #             # 翻頁邏輯
    #             try:
    #                 next_page_button = self.wait.until(EC.element_to_be_clickable(self.SELECTORS["next_page_button"]))
    #                 self.driver.execute_script("arguments[0].click();", next_page_button)
    #                 self.wait.until(EC.invisibility_of_element_located(self.SELECTORS["loading_cover"]))
    #             except (TimeoutException, NoSuchElementException):
    #                 print("未找到『下一页』按鈕，爬取結束。")
    #                 break
    #
    #     except Exception as e:
    #         print(f"發生嚴重錯誤，程式終止: {e}")
    #         self.driver.save_screenshot('fatal_error.png')
    #     finally:
    #         print("正在關閉 WebDriver...")
    #         if self.driver:
    #             self.driver.quit()
    #
    #     # 最後再完整保存一次
    #     df_final = pd.DataFrame(all_products)
    #     df_final.to_csv(output_csv, index=False, encoding='utf-8-sig')
    #     print(f"\n爬取完成！總共獲取 {len(all_products)} 條產品資訊，已保存到 {output_csv}。")
    #     return all_products

    def scrape_category(self, start_url: str, filters: dict, max_pages: int = 5):
        """
        爬取一個指定類別的完整流程。
        """
        self._initialize_driver()
        all_products = []

        try:
            self.driver.get(start_url)
            self._handle_cookies()
            if filters:
                self.apply_filters(filters)

            for page_num in range(1, max_pages + 1):
                print(f"\n--- 正在處理第 {page_num} 頁 ---")

                self.wait.until(EC.presence_of_element_located(self.SELECTORS["product_rows"]))
                time.sleep(random.uniform(2, 4))

                try:
                    indicator_element = self.driver.find_element(*self.SELECTORS["pagination_indicator"])
                    old_indicator_text = indicator_element.text
                    print(f"當前頁面指示器: {old_indicator_text}")
                except NoSuchElementException:
                    print("未找到分頁指示器。")
                    old_indicator_text = "" # 如果找不到，給一個空字串


                page_source = self.driver.page_source
                products_on_page = self._parse_page(page_source)
                all_products.extend(products_on_page)
                print(f"在本頁找到 {len(products_on_page)} 條產品資訊。")

                try:
                    # next_page_button = self.wait.until(EC.element_to_be_clickable(self.SELECTORS["next_page_button"]))
                    # print("找到『下一页』按鈕，正在點擊...")
                    # self.wait.until(
                    #     EC.invisibility_of_element_located(self.SELECTORS["loading_cover"])
                    # )
                    # time.sleep(random.uniform(2, 4))
                    # self.wait.until(
                    #     EC.invisibility_of_element_located(self.SELECTORS["loading_cover"])
                    # )
                    # print("等待完毕")
                    # next_page_button.click()
                    # print("点击")
                    # time.sleep(random.uniform(2, 4))
                    # self.wait.until(
                    #     EC.invisibility_of_element_located(self.SELECTORS["loading_cover"])
                    # )
                    # print("點擊了")

                    # 檢查是否為最後一頁 (如果 "下一页" 按鈕不可點擊)
                    next_page_button = self.driver.find_element(*self.SELECTORS["next_page_button"])
                    if not next_page_button.is_enabled():
                        print("『下一页』按鈕已禁用，爬取結束。")
                        break

                    print("找到『下一页』按鈕，正在點擊...")
                    self.driver.execute_script("arguments[0].click();", next_page_button)

                    # --- 關鍵修正 2：等待「分頁指示器」的文字發生變化 ---
                    print("已點擊翻頁，正在等待內容刷新（等待分頁指示器文字變化）...")
                    self.wait.until(
                        text_to_be_different_from(self.SELECTORS["pagination_indicator"], old_indicator_text)
                    )
                    print("分頁指示器已更新，頁面已成功刷新！")

                except (TimeoutException, NoSuchElementException):
                    print("未找到可點擊的『下一页』按鈕，爬取結束。")
                    break

        except Exception as e:
            print(f"發生嚴重錯誤: {e}")
            self.driver.save_screenshot('debug_error.png')
        finally:
            print("正在關閉 WebDriver...")
            self.driver.quit()

        print(f"\n爬取完成！總共獲取 {len(all_products)} 條產品資訊。")
        return all_products


if __name__ == '__main__':
    DRIVER_PATH = r'D:\PhD\desktop\PhDresearch\3rddigitaltwin\3_Simulation\component_model_framework\database\capacitor\dataset_embedding\data\find_price\chromedriver\chromedriver-win64\chromedriver-win64\chromedriver.exe'
    Ecap_type={
    'Al_elec':"https://www.digikey.cn/zh/products/filter/%E9%93%9D%E7%94%B5%E8%A7%A3%E7%94%B5%E5%AE%B9%E5%99%A8/58",
    'Al_poly':'https://www.digikey.cn/zh/products/filter/%E9%93%9D-%E8%81%9A%E5%90%88%E7%89%A9%E7%94%B5%E5%AE%B9%E5%99%A8/69',
    'Titan_poly':'https://www.digikey.cn/zh/products/filter/%E9%92%BD-%E8%81%9A%E5%90%88%E7%89%A9%E7%94%B5%E5%AE%B9%E5%99%A8/70',
    'Titan':'https://www.digikey.cn/zh/products/filter/%E9%92%BD%E7%94%B5%E5%AE%B9%E5%99%A8/59',
    'film':'https://www.digikey.cn/zh/products/filter/%E8%96%84%E8%86%9C%E7%94%B5%E5%AE%B9%E5%99%A8/62'}
    START_URL = Ecap_type['film']

    FILTERS_TO_APPLY = {
        '制造商': 'EPCOS - TDK Electronics'#'KEMET'
    }

    MAX_PAGES_TO_SCRAPE = 1000  # 測試時設為較小的值

    # 建立爬蟲實例並執行
    scraper = DigiKeyScraper(driver_path=DRIVER_PATH)
    scraped_data = scraper.scrape_category(
        start_url=START_URL,
        filters=FILTERS_TO_APPLY,
        max_pages=MAX_PAGES_TO_SCRAPE
    )

    if scraped_data:
        df = pd.DataFrame(scraped_data)
        df.to_csv("digikey_kemet_capacitors.csv", index=False, encoding='utf-8-sig')
        print("\n數據已成功保存到 digikey_kemet_capacitors.csv")