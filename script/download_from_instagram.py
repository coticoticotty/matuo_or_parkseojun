from pathlib import Path
import sys
import os
import instaloader
import yaml


def main():
    script_dir = Path(__file__).resolve().parent
    yaml_file_path = os.path.join(script_dir, 'scraping.yaml')
    with open(yaml_file_path, 'r') as f:
        data = yaml.safe_load(f)

    USER_ID = data['instagram']['my_account']['user_id']
    PASSWORD = data['instagram']['my_account']['passward']
    TARGET_ID = data['instagram']['scraping_target']['target_account_id']
    HASHTAG = data['instagram']['scraping_target']['target_hashtag']
    NUM_OF_PICTURE = data['instagram']['scraping_target']['num_of_picture']


    cls = GetImageFromInstagram(USER_ID, PASSWORD)
    cls.download_user_posts(TARGET_ID, NUM_OF_PICTURE)

class GetImageFromInstagram():
    def __init__(self, my_user_name, password):
        self.L = instaloader.Instaloader()
        self.my_user_name = my_user_name
        self.password = password
        self.L.login(my_user_name, password)

    def download_user_posts(self, target_username, limitation=100):
        posts = instaloader.Profile.from_username(self.L.context, target_username).get_posts()

        for index, post in enumerate(posts, 1):
            self.L.download_post(post, target_username)
            if index >= limitation:
                print(f'{target_username}の画像を{limitation}枚保存しました。')
                break

    def download_hashtag_posts(self, hashtag, limitation=100):
        posts = instaloader.Hashtag.from_name(self.L.context, hashtag).get_posts()
        for index, post in enumerate(posts, 1):
            self.L.download_post(post, target='#' + hashtag)
            if index >= limitation:
                print(f'ハッシュタグ#{hashtag}の画像を{limitation}枚保存しました。')
                break

if __name__ == '__main__':
    main()