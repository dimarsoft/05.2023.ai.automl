import gdown
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import autokeras as ak
import pprint


def read_dataset() -> DataFrame:
    """
    Считать данные их файла
    :return:
        DataFrame: Данные
    """
    gdown.download('https://storage.yandexcloud.net/aiueducation/Content/base/l10/basketball.csv',
                   ".", quiet=True)
    # Загружаем базу
    df = pd.read_csv('basketball.csv', encoding='cp1251', sep=';', header=0, index_col=0)
    print(df.head()["info"])

    #  df.to_csv('basketball_utf8.csv', encoding='utf-8', sep=';', index=False)

    #  df.head()["info"].to_csv('basketball_info_head_utf8.csv', encoding='utf-8', sep=';',
    #  index=False)

    return df


def split_dataset(df: DataFrame) -> tuple[DataFrame, DataFrame, np.ndarray]:
    """
    Подготовка данных.
    Разделить на данные, текстовые данные, ответы

    :param df: Датафрейм со всеми данными
    :return:
        tuple[DataFrame данные, DataFrame текст, np.ndarray ответы]:

    """
    # ftime содержит "Минута", "Общая минута", "Секунда"
    # удаляем указанные столбцы
    data = df.drop(["info", "fcount", "Минута", "Общая минута", "Секунда"], axis=1)
    data = data.replace(',', '.', regex=True).astype(float)
    data = data.astype(float)

    df_text = df["info"].values
    y_train = np.array(df['fcount'].astype('int'))

    return data, df_text, y_train


def scale_data(data: DataFrame):
    """
    Масштабируем данные, возможно это лучше, чем без него.
    :param data:
    :return:
    """
    scaler = StandardScaler()  # MinMaxScaler()

    # Нормализация значений столбцов 'TOTAL' и 'ftime' с помощью объекта scaler
    data[['TOTAL', 'ftime']] = scaler.fit_transform(data[['TOTAL', 'ftime']])


def check_data(data: DataFrame) -> bool:
    """
    Проверка данных.
    Не должно быть nans
    :param data:
    :return:
    """
    if data.isna().any().any():
        print("Найдены отсутствующие значения в DataFrame")
        return False
    else:
        print("Отсутствующих значений в DataFrame не найдено")
        return True


def train(train_data: DataFrame, txt_data: DataFrame, y_train: np.ndarray) -> ak.AutoModel:
    """
    Получаем модель НС с помощью autokeras
    :param train_data:
    :param txt_data:
    :param y_train:
    :return:
    """
    # Разбиваем по выборкам
    x_train, x_test, data_text_train, data_text_test, y_train, y_test = train_test_split(
        train_data, txt_data, y_train, test_size=7450, shuffle=False)

    # Инициализация модели с несколькими входными и выходными данными.
    model = ak.AutoModel(
        inputs=[ak.TextInput(), ak.StructuredDataInput()],
        outputs=[
            ak.RegressionHead(loss="MAE", metrics=["mae"])
        ],
        overwrite=True,
        max_trials=16,
    )
    # Обучаем модель на подготовленных данных.
    model.fit(
        [data_text_train, x_train],
        y_train,
        epochs=10,
        validation_data=([data_text_test, x_test], y_test)
    )
    return model


def print_model(model: ak.AutoModel):
    """
    Вывод информации о полученной модели НС
    :param model: Модель НС.
    :return:
    """
    best_model = model.export_model()
    # Получите архитектуру модели в виде JSON строки
    model_json = best_model.to_json()

    pprint.pprint(model_json)
    best_model.summary()


def make_train():
    # читаем данные из файла
    df_from_file = read_dataset()

    # делим данные
    train_data, txt_data, y_train = split_dataset(df_from_file)

    # масштабируем данные
    scale_data(train_data)

    print(train_data, txt_data, y_train)

    # проверяем данные

    check_data(train_data)

    # получаем модель

    model = train(train_data, txt_data, y_train)

    # выводим информацию о полученной модели

    print_model(model)


make_train()
