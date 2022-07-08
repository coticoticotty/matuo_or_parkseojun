# おそらくライブラリのバグによりハッシュタグ検索ができない。。。
import instaloader


def main():
    # login using the credentials
    USER = '*************'
    PASSWORD = '***********'

    cls = GetImageFromInstagram(USER, PASSWORD)
    cls.download_user_posts('matsuoshun', 300)

    # パク・ソジュンの画像を入手
    cls.download_user_posts('parkseojun', 300)

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