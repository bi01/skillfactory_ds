import numpy as np


def game_core_v1(number):
    """ Просто угадываем на random, никак не используя информацию о больше или меньше.
        Функция принимает загаданное число и возвращает число попыток
    """

    counter = 0
    while True:
        counter += 1
        predict_number = np.random.randint(1, 101)  # предполагаемое число
        if number == predict_number:
            return counter  # выход из цикла, если угадали


def game_core_v2(number):
    """ Сначала устанавливаем любое random число,
        а потом уменьшаем или увеличиваем его в зависимости от того,
        больше оно или меньше нужного.
        Функция принимает загаданное число и возвращает число попыток
    """

    counter = 1
    predict = np.random.randint(1, 101)

    while number != predict:
        counter += 1
        if number > predict:
            predict += 1
        elif number < predict:
            predict -= 1

    return counter  # выход из цикла, если угадали


def my_game_core(hidden_number: int, start=1, end=101) -> int:
    """ Ищет загаданное число, беря среднее значение между 2-мя крайними.
        По-другому называется двоичный или бинарный поиск
    """

    def mean_between(num1: int, num2: int) -> int:
        return num1 + (num2 - num1) // 2

    predict_number = mean_between(start, end)
    counter: int = 1

    while hidden_number != predict_number:
        if hidden_number > predict_number:
            start = predict_number

        elif hidden_number < predict_number:
            end = predict_number

        predict_number = mean_between(start, end)
        counter += 1

    return counter


def score_game(game_core) -> int:
    """ Запускаем игру 1000 раз, чтобы узнать, как быстро игра угадывает число
    """

    # фиксируем RANDOM SEED, чтобы ваш эксперимент был воспроизводим!
    np.random.seed(1)

    random_array = np.random.randint(1, 101, size=1000)

    count_ls = [game_core(item) for item in random_array]
    score = int(np.mean(count_ls))
    print(f"Ваш алгоритм угадывает число в среднем за {score} попыток")

    return score


if __name__ == "__main__":

    # Проверяем
    score_game(game_core_v2)
    score_game(game_core_v1)
    score_game(my_game_core)
