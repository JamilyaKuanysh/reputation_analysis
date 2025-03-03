
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
import os
import ssl
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# Обход проблемы с SSL для загрузки NLTK данных
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Загрузка необходимых ресурсов NLTK
print("Загрузка ресурсов NLTK...")
nltk.download('punkt')
nltk.download('stopwords')

print("=" * 80)
print("РЕШЕНИЕ: МОНИТОРИНГ И АНАЛИЗ РЕПУТАЦИИ КОМПАНИИ В СМИ")
print("=" * 80)


def load_newscatcher_dataset(filepath):
    """
    Загрузка и анализ датасета newscatcher с различными параметрами

    Args:
        filepath (str): Путь к файлу CSV

    Returns:
        pd.DataFrame: Подготовленный DataFrame
    """
    print(f"\nЗагрузка данных из файла: {filepath}")

    # Проверка существования файла
    if not os.path.exists(filepath):
        print(f"ОШИБКА: Файл {filepath} не найден!")
        return None

    try:
        # Попробуем сначала посмотреть первые несколько строк файла
        with open(filepath, 'r', encoding='utf-8') as f:
            first_lines = [next(f) for _ in range(5)]

        print("Первые 5 строк файла:")
        for i, line in enumerate(first_lines):
            print(f"Строка {i + 1}: {line[:100]}...")

        # Используем точку с запятой как разделитель
        print("Используем разделитель: ';'")

        # Список кодировок для попытки загрузки
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

        # Попытка загрузки с разными кодировками и настройками
        for encoding in encodings:
            try:
                # Загрузка с разделителем точка с запятой и явным указанием столбцов
                df = pd.read_csv(filepath,
                                 sep=';',
                                 encoding=encoding,
                                 on_bad_lines='skip',
                                 names=['topic', 'link', 'domain', 'published_date', 'title', 'lang'],
                                 header=0)  # Указываем, что первая строка - заголовок

                if not df.empty:
                    print(f"Файл успешно загружен с кодировкой {encoding}.")
                    print(f"Размер: {df.shape[0]} строк, {df.shape[1]} колонок")

                    # Вывод названий колонок
                    print("\nКолонки в датасете:")
                    print(df.columns.tolist())

                    # Базовая очистка данных
                    df = df.dropna(subset=['title'])  # Удаляем строки без заголовка
                    df = df.reset_index(drop=True)  # Сбрасываем индекс

                    return df

            except Exception as e:
                print(f"Не удалось загрузить с кодировкой {encoding}: {str(e)}")

        # Если все попытки провалились
        print("ОШИБКА: Не удалось загрузить данные ни с одной из кодировок.")
        return None

    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА при загрузке данных: {str(e)}")
        return None


def main():
    # Получаем путь к текущей директории скрипта
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Путь к файлу в той же директории, что и скрипт
    filepath = os.path.join(base_dir, 'labelled_newscatcher_dataset.csv')

    # Загрузка данных
    df = load_newscatcher_dataset(filepath)

    if df is not None:
        # Здесь будет ваш дальнейший код анализа
        print("\nДанные успешно загружены. Продолжение анализа...")

        # Например, первичный осмотр данных
        print("\nПервичный осмотр данных:")
        print(df.info())

        # Проверка уникальных значений
        if 'topic' in df.columns:
            print("\nУникальные темы:")
            print(df['topic'].unique())


if __name__ == '__main__':
    main()
# 3. АНАЛИЗ ТОНАЛЬНОСТИ
# -----------------------------------------

class SentimentAnalyzer:
    """
    Класс для анализа тональности новостей
    """

    def __init__(self):
        """
        Инициализация анализатора тональности
        """
        print("Инициализация анализатора тональности...")
        self.preprocessor = TextPreprocessor(language='english')
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.classifier = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, class_weight='balanced')
        self.classes = ['negative', 'neutral', 'positive']
        self.trained = False

    def train(self, texts, labels):
        """
        Обучение модели анализа тональности

        Args:
            texts (list): Список текстов
            labels (list): Список меток тональности

        Returns:
            self: Объект SentimentAnalyzer
        """
        print(f"Обучение модели анализа тональности на {len(texts)} текстах...")

        if len(texts) == 0 or len(labels) == 0:
            print("ОШИБКА: Пустые данные для обучения")
            return self

        # Предобработка текстов
        print("Предобработка текстов для обучения...")
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]

        # Статистика по меткам
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        print("Распределение меток для обучения:")
        for label, count in label_counts.items():
            print(f"- {label}: {count} ({count / len(labels) * 100:.1f}%)")

        # Векторизация и обучение
        print("Векторизация текстов...")
        X = self.vectorizer.fit_transform(processed_texts)
        print(f"Размерность векторного представления: {X.shape}")

        print("Обучение классификатора...")
        self.classifier.fit(X, labels)
        self.trained = True

        # Оценка на обучающих данных
        y_pred = self.classifier.predict(X)
        accuracy = accuracy_score(labels, y_pred)
        print(f"Точность на обучающих данных: {accuracy:.2f}")

        print("Обучение модели анализа тональности успешно завершено")
        return self

    def predict(self, texts):
        """
        Предсказание тональности для новых текстов

        Args:
            texts (list): Список текстов

        Returns:
            list: Список предсказанных тональностей
        """
        print(f"Анализ тональности для {len(texts)} текстов...")

        if not self.trained:
            print("ПРЕДУПРЕЖДЕНИЕ: Модель не обучена, используем правила для определения тональности")
            return self._simple_rule_based_sentiment(texts)

        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        X = self.vectorizer.transform(processed_texts)
        predictions = self.classifier.predict(X)

        # Статистика предсказаний
        pred_counts = {}
        for pred in predictions:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1

        print("Результаты анализа тональности:")
        for pred, count in pred_counts.items():
            print(f"- {pred}: {count} ({count / len(predictions) * 100:.1f}%)")

        return predictions

    def _simple_rule_based_sentiment(self, texts):
        """
        Простой анализ тональности на основе правил для английского языка

        Args:
            texts (list): Список текстов

        Returns:
            list: Список предсказанных тональностей
        """
        print("Применение правил для определения тональности...")

        results = []
        for text in texts:
            if not isinstance(text, str):
                results.append('neutral')
                continue

            text = text.lower()

            # Позитивные и негативные маркеры для английского языка
            positive_markers = ['success', 'growth', 'increase', 'profit', 'award',
                                'recognition', 'innovation', 'leader', 'improvement', 'development',
                                'rise', 'excellent', 'advantage', 'benefit', 'win', 'positive']

            negative_markers = ['problem', 'decline', 'decrease', 'loss', 'complaint',
                                'violation', 'legal', 'scandal', 'fine', 'closure', 'fail',
                                'crisis', 'risk', 'threat', 'poor', 'negative', 'bad']

            pos_count = sum(1 for word in positive_markers if word in text)
            neg_count = sum(1 for word in negative_markers if word in text)

            if pos_count > neg_count:
                results.append('positive')
            elif neg_count > pos_count:
                results.append('negative')
            else:
                results.append('neutral')

        # Статистика результатов
        result_counts = {}
        for res in results:
            result_counts[res] = result_counts.get(res, 0) + 1

        print("Результаты анализа тональности на основе правил:")
        for res, count in result_counts.items():
            print(f"- {res}: {count} ({count / len(results) * 100:.1f}%)")

        return results


# -----------------------------------------
# 4. ФУНКЦИЯ АНАЛИЗА ТОНАЛЬНОСТИ ПУБЛИКАЦИЙ
# -----------------------------------------

def analyze_sentiment(df, text_column='title'):
    """
    Анализ тональности публикаций

    Args:
        df (pd.DataFrame): Датафрейм с данными
        text_column (str): Колонка с текстом для анализа

    Returns:
        tuple: Анализатор тональности и обновленный датафрейм
    """
    print("\n" + "=" * 50)
    print("АНАЛИЗ ТОНАЛЬНОСТИ ПУБЛИКАЦИЙ")
    print("=" * 50)

    if text_column not in df.columns:
        print(f"ОШИБКА: Колонка '{text_column}' не найдена в данных")
        if 'title' in df.columns:
            text_column = 'title'
            print(f"Используем колонку 'title' для анализа")
        else:
            print("Нет подходящей колонки для анализа тональности")
            return None, df

    # Предобработка текста
    print("\n1. Предобработка текста")
    preprocessor = TextPreprocessor(language='english')
    df = preprocessor.preprocess_dataframe(df, text_column, f'processed_{text_column}')

    # Подготовка данных для обучения
    if 'sentiment' in df.columns and df['sentiment'].nunique() > 1:
        print("\n2. Подготовка данных с имеющимися метками тональности")
        print(f"Имеется {df['sentiment'].nunique()} уникальных значений тональности")
    else:
        print("\n2. Создание синтетических меток тональности")
        # Создаем простой анализатор для определения начальных меток
        temp_analyzer = SentimentAnalyzer()
        df['sentiment'] = temp_analyzer._simple_rule_based_sentiment(df[text_column].tolist())

    # Разделение на обучающую и тестовую выборки
    print("\n3. Разделение данных для обучения и оценки")

    # Сбалансированная выборка для обучения
    train_df = pd.DataFrame()
    for sentiment in df['sentiment'].unique():
        sentiment_df = df[df['sentiment'] == sentiment]
        sample_size = min(len(sentiment_df), 200)  # Ограничиваем для демонстрации
        if sample_size > 0:
            train_subset = sentiment_df.sample(n=sample_size, random_state=42)
            train_df = pd.concat([train_df, train_subset])

    print(f"Создана обучающая выборка размером {len(train_df)} записей")

    # Выбираем тестовую выборку из оставшихся данных
    test_df = df[~df.index.isin(train_df.index)]
    test_size = min(len(test_df), 300)
    if test_size > 0:
        test_df = test_df.sample(n=test_size, random_state=42)
    else:
        test_df = train_df.sample(frac=0.3, random_state=42)

    print(f"Создана тестовая выборка размером {len(test_df)} записей")

    # Обучение модели
    print("\n4. Обучение модели анализа тональности")
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_analyzer.train(train_df[text_column].tolist(), train_df['sentiment'].tolist())

    # Оценка модели
    print("\n5. Оценка качества модели")
    predicted = sentiment_analyzer.predict(test_df[text_column].tolist())

    print("\nМатрица ошибок:")
    cm = confusion_matrix(test_df['sentiment'].tolist(), predicted)
    print(cm)

    print("\nОтчет о классификации:")
    print(classification_report(test_df['sentiment'].tolist(), predicted))

    # Визуализация матрицы ошибок
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(test_df['sentiment']),
                yticklabels=np.unique(test_df['sentiment']))
    plt.xlabel('Предсказанная тональность')
    plt.ylabel('Истинная тональность')
    plt.title('Матрица ошибок модели анализа тональности')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Матрица ошибок сохранена в файл confusion_matrix.png")

    # Применение модели ко всему датасету
    print("\n6. Применение модели ко всем данным")
    df['predicted_sentiment'] = sentiment_analyzer.predict(df[text_column].tolist())

    # Анализ распределения тональности
    print("\n7. Анализ распределения тональности")
    sentiment_counts = df['predicted_sentiment'].value_counts()
    print("Распределение тональности:")
    for sentiment, count in sentiment_counts.items():
        print(f"- {sentiment}: {count} ({count / len(df) * 100:.1f}%)")

    # Визуализация распределения тональности
    plt.figure(figsize=(10, 6))
    colors = {'positive': 'green', 'neutral': 'blue', 'negative': 'red'}
    palette = {s: colors.get(s, 'gray') for s in df['predicted_sentiment'].unique()}
    ax = sns.countplot(x='predicted_sentiment', data=df, palette=palette)
    plt.title('Распределение тональности публикаций')
    plt.xlabel('Тональность')
    plt.ylabel('Количество публикаций')

    # Добавляем подписи значений на столбцы
    for i, count in enumerate(sentiment_counts):
        ax.text(i, count + 5, str(count), ha='center')

    plt.tight_layout()
    plt.savefig('sentiment_distribution.png')
    print("График распределения тональности сохранен в файл sentiment_distribution.png")

    # Анализ тональности по источникам
    if 'source' in df.columns:
        print("\n8. Анализ тональности по источникам")

        # Выбираем топ-10 источников по количеству публикаций
        top_sources = df['source'].value_counts().head(10).index
        source_df = df[df['source'].isin(top_sources)]

        if len(source_df) > 0:
            source_sentiment = pd.crosstab(source_df['source'], source_df['predicted_sentiment'])
            print("\nРаспределение тональности по топ-10 источникам:")
            print(source_sentiment)

            # Визуализация
            plt.figure(figsize=(12, 8))
            source_sentiment.plot(kind='bar', stacked=True, colormap='viridis')
            plt.title('Распределение тональности по топ-10 источникам')
            plt.xlabel('Источник')
            plt.ylabel('Количество публикаций')
            plt.legend(title='Тональность')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('source_sentiment.png')
            print("График распределения тональности по источникам сохранен в source_sentiment.png")

    # Анализ тональности по времени, если есть данные о датах
    if 'date' in df.columns:
        print("\n9. Анализ динамики тональности во времени")

        # Группировка по месяцам
        try:
            df['month'] = df['date'].dt.to_period('M')
            monthly_sentiment = pd.crosstab(df['month'], df['predicted_sentiment'])

            if len(monthly_sentiment) > 1:  # Проверка, есть ли данные за несколько месяцев
                print("\nДинамика тональности по месяцам:")
                print(monthly_sentiment)

                # Визуализация динамики
                plt.figure(figsize=(14, 7))
                monthly_sentiment.plot(kind='line', marker='o')
                plt.title('Динамика тональности публикаций по месяцам')
                plt.xlabel('Месяц')
                plt.ylabel('Количество публикаций')
                plt.legend(title='Тональность')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig('sentiment_dynamics.png')
                print("График динамики тональности сохранен в файл sentiment_dynamics.png")

                # Добавляем индекс репутации (от -100 до 100)
                monthly_sentiment['reputation_index'] = (
                                                                monthly_sentiment.get('positive',
                                                                                      0) - monthly_sentiment.get(
                                                            'negative', 0)
                                                        ) / monthly_sentiment.sum(axis=1) * 100

                print("\nИндекс репутации по месяцам:")
                print(monthly_sentiment['reputation_index'])

                plt.figure(figsize=(14, 7))
                monthly_sentiment['reputation_index'].plot(kind='line', marker='o', color='purple')
                plt.title('Индекс репутации по месяцам')
                plt.xlabel('Месяц')
                plt.ylabel('Индекс репутации (-100 до 100)')
                plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig('reputation_index.png')
                print("График индекса репутации сохранен в файл reputation_index.png")
        except Exception as e:
            print(f"ОШИБКА при анализе динамики тональности: {str(e)}")

    print("\nАнализ тональности успешно завершен")
    return sentiment_analyzer, df


# -----------------------------------------
# ЗАПУСК ПОЛНОГО РЕШЕНИЯ
# -----------------------------------------
class TextPreprocessor:
    """
    Класс для предобработки текста новостей
    """

    def __init__(self, language='english'):
        """
        Инициализация препроцессора

        Args:
            language (str): Язык для обработки
        """
        print(f"Инициализация препроцессора текста для языка '{language}'...")
        self.language = language

        try:
            self.stop_words = set(stopwords.words(language))
            print(f"Загружено {len(self.stop_words)} стоп-слов")
        except Exception as e:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Ошибка при загрузке стоп-слов: {str(e)}")
            self.stop_words = set(
                ['a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'at', 'from', 'by', 'for', 'with',
                 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
                 'of', 'in', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
                 'just', 'don', 'should', 'now'])

        try:
            self.stemmer = SnowballStemmer(language)
            print("Стеммер успешно инициализирован")
        except Exception as e:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Ошибка при инициализации стеммера: {str(e)}")
            self.stemmer = None

        # Дополнительные стоп-слова для английского языка
        if language == 'english':
            self.stop_words.update(['this', 'that', 'these', 'those', 'would', 'could', 'should', 'company', 'said'])

    def preprocess(self, text, stem=True):
        """
        Предобработка текста

        Args:
            text (str): Исходный текст
            stem (bool): Применять ли стемминг

        Returns:
            str: Обработанный текст
        """
        if not isinstance(text, str) or not text:
            return ""

        # Приведение к нижнему регистру
        text = text.lower()

        # Удаление специальных символов и цифр
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)

        # Токенизация
        try:
            tokens = word_tokenize(text)
        except:
            # Если NLTK токенизатор не работает, используем простой сплит
            tokens = text.split()

        # Удаление стоп-слов и коротких слов
        filtered_tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]

        # Стемминг при необходимости
        if stem and self.stemmer:
            try:
                filtered_tokens = [self.stemmer.stem(token) for token in filtered_tokens]
            except Exception as e:
                print(f"ПРЕДУПРЕЖДЕНИЕ: Ошибка при стемминге: {str(e)}")

        return ' '.join(filtered_tokens)

    def preprocess_dataframe(self, df, text_column, new_column=None, stem=True):
        """
        Предобработка колонки с текстом в DataFrame

        Args:
            df (pd.DataFrame): Исходный DataFrame
            text_column (str): Имя колонки с текстом
            new_column (str, optional): Имя новой колонки
            stem (bool): Применять ли стемминг

        Returns:
            pd.DataFrame: DataFrame с обработанным текстом
        """
        print(f"Предобработка текста из колонки '{text_column}'...")

        if text_column not in df.columns:
            print(f"ОШИБКА: Колонка '{text_column}' не найдена в DataFrame")
            return df

        if new_column is None:
            new_column = f'processed_{text_column}'

        # Обработка небольшой порции для теста
        sample = df[text_column].iloc[0] if len(df) > 0 else ""
        processed_sample = self.preprocess(sample, stem=stem)
        print(f"Пример обработки текста:")
        print(f"Оригинал: {sample[:100]}...")
        print(f"После обработки: {processed_sample[:100]}...")

        # Обработка всего DataFrame
        print(f"Обработка {len(df)} текстов...")
        df[new_column] = df[text_column].apply(lambda x: self.preprocess(x, stem=stem))
        print(f"Обработка текста завершена. Результаты сохранены в колонке '{new_column}'")

        return df
def run_analysis():
    """
    Запуск полного анализа репутации
    """
    print("\n" + "=" * 80)
    print("ЗАПУСК АНАЛИЗА РЕПУТАЦИИ С РЕАЛЬНЫМ ДАТАСЕТОМ")
    print("=" * 80)

    # 1. Загрузка и анализ данных
    print("\nШАГ 1: Загрузка и анализ данных")
    df = load_newscatcher_dataset("labelled_newscatcher_dataset.csv")

    if df is None or len(df) == 0:
        print("\nОШИБКА: Не удалось загрузить данные. Завершение программы.")
        return

    # 2. Анализ тональности
    print("\nШАГ 2: Анализ тональности публикаций")
    sentiment_analyzer, df_with_sentiment = analyze_sentiment(df)

    # 3. Сохранение результатов
    print("\nШАГ 3: Сохранение результатов анализа")
    try:
        output_file = "processed_newscatcher_data.csv"
        df_with_sentiment.to_csv(output_file, index=False)
        print(f"Результаты анализа сохранены в файл {output_file}")
        print(f"Размер сохраненного файла: {os.path.getsize(output_file) / 1024:.1f} KB")
    except Exception as e:
        print(f"ОШИБКА при сохранении результатов: {str(e)}")

    # 4. Выводы и рекомендации
    print("\n" + "=" * 80)
    print("ВЫВОДЫ И РЕКОМЕНДАЦИИ")
    print("=" * 80)

    # Анализ результатов тональности
    sentiment_counts = df_with_sentiment['predicted_sentiment'].value_counts()
    total_articles = len(df_with_sentiment)

    # Получаем процентное соотношение с проверкой на наличие меток
    positive_pct = sentiment_counts.get('positive', 0) / total_articles * 100
    negative_pct = sentiment_counts.get('negative', 0) / total_articles * 100
    neutral_pct = sentiment_counts.get('neutral', 0) / total_articles * 100

    # Индекс репутации (от -100 до 100)
    reputation_index = positive_pct - negative_pct

    print(f"\nАНАЛИЗ ТЕКУЩЕЙ РЕПУТАЦИИ:")
    print(f"Проанализировано публикаций: {total_articles}")
    print(f"Распределение тональности: ")
    print(f"- Позитивные: {positive_pct:.1f}%")
    print(f"- Нейтральные: {neutral_pct:.1f}%")
    print(f"- Негативные: {negative_pct:.1f}%")
    print(f"Индекс репутации: {reputation_index:.1f} (от -100 до 100)")

    # Определение статуса репутации
    if reputation_index >= 30:
        reputation_status = "Отличная"
    elif reputation_index >= 10:
        reputation_status = "Хорошая"
    elif reputation_index >= -10:
        reputation_status = "Нейтральная"
    elif reputation_index >= -30:
        reputation_status = "Требует внимания"
    else:
        reputation_status = "Критическая"

    print(f"Статус репутации: {reputation_status}")

    # Анализ тематики публикаций
    if 'topic' in df_with_sentiment.columns:
        print("\nАнализ тематики публикаций с разбивкой по тональности:")
        topic_sentiment = pd.crosstab(df_with_sentiment['topic'], df_with_sentiment['predicted_sentiment'])
        topic_sentiment_pct = pd.crosstab(df_with_sentiment['topic'], df_with_sentiment['predicted_sentiment'],
                                          normalize='index') * 100

        for topic in topic_sentiment.index:
            positive = topic_sentiment_pct.loc[topic, 'positive'] if 'positive' in topic_sentiment_pct.columns else 0
            negative = topic_sentiment_pct.loc[topic, 'negative'] if 'negative' in topic_sentiment_pct.columns else 0

            topic_reputation = positive - negative

            status = "Отличная" if topic_reputation >= 30 else \
                "Хорошая" if topic_reputation >= 10 else \
                    "Нейтральная" if topic_reputation >= -10 else \
                        "Требует внимания" if topic_reputation >= -30 else \
                            "Критическая"

            print(f"- Тема '{topic}': {topic_reputation:.1f} ({status})")

    # Рекомендации на основе анализа
    print("\nКЛЮЧЕВЫЕ РЕКОМЕНДАЦИИ:")

    if reputation_index < 0:
        print("1. Разработать антикризисную коммуникационную стратегию для улучшения репутации")
        print("2. Активно работать с негативными публикациями через официальные каналы коммуникации")
        print("3. Увеличить частоту позитивных информационных поводов")
    else:
        print("1. Поддерживать текущую коммуникационную стратегию с акцентом на сильные стороны")
        print("2. Развивать присутствие в ключевых СМИ с высоким охватом целевой аудитории")
        print("3. Использовать успешные кейсы и достижения для создания позитивных информационных поводов")

    print("\n4. Внедрить автоматизированную систему мониторинга репутации для оперативного реагирования")
    print("5. Разработать KPI для оценки эффективности управления репутацией")

    # Заключение
    print("\nЗАКЛЮЧЕНИЕ:")
    print("Внедрение системы мониторинга и анализа репутации с использованием ML/NLP позволит:")
    print("- Оперативно выявлять репутационные риски и своевременно на них реагировать")
    print("- Отслеживать эффективность PR и коммуникационных стратегий")
    print("- Получать объективные данные для принятия решений в области управления репутацией")
    print("- Прогнозировать развитие информационных ситуаций и их влияние на бизнес")

    print("\nПредложенный прототип демонстрирует возможности применения ML/NLP для мониторинга и анализа репутации.")
    print("Решение может быть масштабировано и доработано в соответствии с конкретными потребностями бизнеса.")

    return sentiment_analyzer, df_with_sentiment


# Главная функция для запуска анализа
if __name__ == "__main__":
    run_analysis()



