from random import randrange, choice
import numpy as np


# Создаёт тестовые данные для модели
class WeatherGenerator:

    # Генерирует случайную запись о погоде
    def generate_data(self):
        return {
            'Date': f"{randrange(2000, 2010)}-{randrange(1, 12)}-{randrange(1, 28)}",
            'Location': choice(['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree']),
            'MinTemp': str(randrange(7, 15)),
            'MaxTemp': str(randrange(8, 18)),
            'Rainfall': str(randrange(5, 10) / 10),
            'Evaporation': str(randrange(0, 5)),
            'Sunshine': str(randrange(0, 8)),
            'WindGustDir': choice(['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE', 'SSE']),
            'WindGustSpeed': str(randrange(0, 40)),
            'WindDir9am': choice(['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE', 'SSE']),
            'WindDir3pm': choice(['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE', 'SSE']),
            'WindSpeed9am': str(randrange(0, 40)),
            'WindSpeed3pm': str(randrange(0, 40)),
            'Humidity9am': str(randrange(0, 80)),
            'Humidity3pm': str(randrange(0, 80)),
            'Pressure9am': str(randrange(980, 1020)),
            'Pressure3pm': str(randrange(980, 1020)),
            'Cloud9am': str(randrange(0, 9)),
            'Cloud3pm': str(randrange(0, 9)),
            'Temp9am': str(randrange(-7, 25)),
            'Temp3pm': str(randrange(0, 35)),
            'RainToday': choice(['No', 'Yes']),
            'RainTomorrow': choice(['0', '1']),
        }

    def create_row(self, sep=','):
        data = self.generate_data()
        title_row = sep.join(data.keys())
        row = sep.join(list(data.values()))
        return [title_row, row]

    # Создаёт датасет из указанного количества строк
    def generate_csv(self, num_elements):
        dataset = []
        for i in range(num_elements):
            title_row, row = self.create_row()
            if len(dataset) == 0:
                dataset.append(title_row)
            dataset.append(row)
        return np.array(dataset)


def create_csv(num_elements = 1000):
    generator = WeatherGenerator()
    dataset = generator.generate_csv(num_elements)
    np.savetxt("ml_project/tests/help_test_data/test_data.csv", dataset, delimiter="\n", fmt='%s')


def main():
    create_csv(1000)
    print('saved')


if __name__ == "__main__":
    main()
