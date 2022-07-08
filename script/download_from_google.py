import requests
import time
import random
import string
import re
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import ElementClickInterceptedException
from bs4 import BeautifulSoup
import os


def collect_images(search_word):
    QUERY = search_word
    LIMIT_DL_NUM = 300
    RETRY_NUM = 3
    DRIVER_PATH = "C:/Users/Kohei Okamoto/Desktop/study/chromedriver"
    TIMEOUT = 3

    # フルスクリーンにする
    options = webdriver.ChromeOptions()
    options.add_argument("--start-fullscreen")
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')  # headlessモードで暫定的に必要なフラグ(そのうち不要になる)
    options.add_argument('--disable-extensions')  # すべての拡張機能を無効にする。ユーザースクリプトも無効にする
    options.add_argument('--proxy-server="direct://"')  # Proxy経由ではなく直接接続する
    options.add_argument('--proxy-bypass-list=*')  # すべてのホスト名
    options.add_argument('--start-maximized')  # 起動時にウィンドウを最大化する

    # 指定したURLに移動
    url = f'https://www.google.com/search?q={QUERY}&tbm=isch'
    driver = webdriver.Chrome(DRIVER_PATH, options=options)

    # タイムアウト設定
    driver.implicitly_wait(TIMEOUT)

    driver.get(url)

    thumbnail_elements = driver.find_elements(By.CLASS_NAME, 'Q4LuWd')

    count = len(thumbnail_elements)
    print(count)

    while count < LIMIT_DL_NUM:
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        time.sleep(2)

        # サムネイル画像の取得
        thumbnail_elements = driver.find_elements(By.CLASS_NAME, 'Q4LuWd')
        count = len(thumbnail_elements)
        print(count)

    # HTTPヘッダーの作成
    HTTP_HEADERS = {'User-Agent': driver.execute_script('return navigator.userAgent;')}
    print(HTTP_HEADERS)

    image_urls = []

    for index, thumbnail_element in enumerate(thumbnail_elements):
        is_clicked = False
        for i in range(RETRY_NUM):
            try:
                if is_clicked == False:
                    thumbnail_element.click()
                    time.sleep(2)
                    is_clicked = True
            except NoSuchElementException:
                print(f'****NoSuchElementException*****')
                continue
            except Exception:
                print('予期せぬエラーです')
                break

            try:
                # qdnLaf isv-id b0vFpe
                image_element = driver.find_element(By.CLASS_NAME, 'KAlRDb')
                # click_element = driver.find_element(By.CLASS_NAME, 'v4dQwb')
                # image_element = click_element.find_element(By.TAG_NAME, 'img')
                image_url = image_element.get_attribute('src')

                if re.match(r'data:image', image_url):  # サムネイル画像のURLのままだった場合
                    print(f'URLが変わるまで待ちましょう。{i+1}回目')
                    time.sleep(2)
                    if i+1 == RETRY_NUM:
                        print(f'urlは変わりませんでしたね。。。')
                    continue
                else:
                    print(f'image_url: {image_url}')
                    extension = get_extension(image_url)
                    print(image_url)

                    if extension:
                        image_urls.append(image_url)
                        print(f'urlの保存に成功')

                    # jpg jpeg png 以外はダウンロードしない
                    else:
                        print('対象の拡張子ではありません')
                    break
            except NoSuchElementException:
                print(f'****NoSuchElementException*****')
                break

            except ElementClickInterceptedException:
                print(f'***** click エラー: {i+1}回目')
                driver.execute_script('arguments[0].scrollIntoView(true);', thumbnail_element)
                time.sleep(1)
            else:
                break

        if index+1 % 20 == 0:
            print(f'{index+1}件完了')
        time.sleep(1)

    # 出力フォルダの作成
    save_dir = f'./{QUERY}'
    os.makedirs(save_dir, exist_ok=True)

    for image_url in image_urls:
        down_load_image(image_url, save_dir, 3, HTTP_HEADERS)

    driver.quit()

def get_extension(url):
    url_lower = url.lower()
    extension = re.search(r'\.jpg|\.jpeg|\.png', url_lower)
    if extension:
        return extension.group()
    else:
        return None

def randomname(n):
   randlst = [random.choice(string.ascii_letters + string.digits) for i in range(n)]
   return ''.join(randlst)

def down_load_image(url, save_dir, loop, http_header):
    result = False
    for i in range(loop):
        try:
            r = requests.get(url, headers=http_header, stream=True, timeout=10)
            r.raise_for_status()

            extension = get_extension(url)
            file_name = randomname(12)
            file_path = save_dir + '/' + file_name + extension

            with open(file_path, 'wb') as f:
                f.write(r.content)

            print(f'{url}の保存に成功')

        except requests.exceptions.SSLError:
            print('*****SSLエラー*****')
            break

        except requests.exceptions.RequestException as e:
            print(f'***** requests エラー ({e}): {i+1} 回目')
            time.sleep(1)
        else:
            result = True
            break
    return result

if __name__ == '__main__':
    collect_images('パクソジュン')
