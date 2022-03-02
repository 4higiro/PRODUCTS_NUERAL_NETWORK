#include <iostream>
#include <math.h>
#include <fstream>

using namespace std;

float sigm(float x)
{
	return 1 / (1 + exp(-x));
}

void forwordPropagation(bool input[2], float association[2], float& output, float weigths_1[2][2], float weigths_2[2], float biases_1[2], float biases_2)
{
	float sum[2];

	for (int i = 0; i < 2; i++)
	{
		sum[i] = 0;

		for (int j = 0; j < 2; j++)
		{
			sum[i] += input[j] * weigths_1[i][j];
		}

		sum[i] += biases_1[i];
	}

	for (int i = 0; i < 2; i++)
	{
		association[i] = sigm(sum[i]);
	}

	output = sigm(association[0] * weigths_2[0] + association[1] * weigths_2[1] + biases_2);
}

void backPropagation(bool input[2], float association[2], float output, float weigths_1[2][2], float weigths_2[2], float biases_1[2], float& biases_2, float p, float grad_1[2][2], float grad_2[2], float grad_b1[2], float& grad_b2)
{
	forwordPropagation(input, association, output, weigths_1, weigths_2, biases_1, biases_2);

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			grad_1[i][j] = 2 * (output - p) * output * (1 - output) * weigths_2[i] * association[i] * (1 - association[i]) * input[j];
		}

		grad_b1[i] = 2 * (output - p) * output * (1 - output) * weigths_2[i] * association[i] * (1 - association[i]);
	}

	for (int i = 0; i < 2; i++)
	{
		grad_2[i] = 2 * (output - p) * output * (1 - output) * association[i];
	}

	grad_b2 = 2 * (output - p) * output * (1 - output);
}

void counstruct(float weigths_1[2][2], float weigths_2[2], float biases_1[2], float& biases_2)
{
	for (int i = 0; i < 2; i++)
	{
		weigths_2[i] = 1;

		biases_1[i] = 0;

		for (int j = 0; j < 2; j++)
		{
			weigths_1[i][j] = 1;
		}
	}

	biases_2 = 0;
}

void learn(float weigths_1[2][2], float grad_1[2][2], float weigths_2[2], float grad_2[2], float biases_1[2], float grad_b1[2], float& biases_2, float grad_b2)
{
	for (int i = 0; i < 2; i++)
	{
		weigths_2[i] -= grad_2[i];

		biases_1[i] -= grad_b1[i];

		for (int j = 0; j < 2; j++)
		{
			weigths_1[i][j] -= grad_1[i][j];
		}
	}

	biases_2 -= grad_b2;
}

void inputData(bool input[2])
{
	cout << "Введите X (1 / 0): ";
	cin >> input[0];

	cout << "Введите Y (1 / 0):";
	cin >> input[1];
}

void printConfig(float weigths_1[2][2], float weigths_2[2], float biases_1[2], float biases_2)
{
	cout << endl << "==============================================================" << endl;

	cout << "Weights 1->2: " << endl;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			cout << weigths_1[i][j] << "\t";
		}

		cout << endl;
	}

	cout << "Biases 1->2: " << endl;

	for (int i = 0; i < 2; i++)
	{
		cout << biases_1[i] << "\t";
	}

	cout << endl;

	cout << "Weights 2->3: " << endl;

	for (int i = 0; i < 2; i++)
	{
		cout << weigths_2[i] << "\t";

		cout << endl;
	}

	cout << "Biases 2->3: " << endl;

	cout << biases_2 << endl;

	cout << endl << "==============================================================" << endl;
}

void printAnswer(float output)
{
	cout << endl << "Ответ нейросети: ";

	if (output > 0.5)
	{
		cout << 1;
	}
	else
	{
		cout << 0;
	}

	cout << " ( " << output << " ) " << endl;
}

void main()
{
	setlocale(LC_ALL, "Ru");

	bool input[2];
	float association[2];
	float output = 0;
	float weigths_1[2][2];
	float weigths_2[2];
	float biases_1[2];
	float biases_2;
	float p;

	float grad_1[2][2];
	float grad_2[2];
	float grad_b1[2];
	float grad_b2;

	counstruct(weigths_1, weigths_2, biases_1, biases_2);

	ifstream file;
	file.open("resources/data.txt");

	if (!file.is_open())
	{
		cout << "-- Файл не найден --" << endl;
	}

	bool data[4][3];

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			file >> data[i][j];
		}
	}

	file.close();

	cout << "Обучение по датасету: " << endl;

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			cout << data[i][j] << "\t";
		}

		cout << endl;
	}

	for (int k = 0; k < 100; k++)
	{
		float av_grad_1[2][2] = { { 0, 0 }, { 0, 0 } };
		float av_grad_2[2] = { 0, 0 };
		float av_grad_b1[2] = { 0, 0 };
		float av_grad_b2 = 0;

		for (int i = 0; i < 4; i++)
		{
			input[0] = data[i][0];
			input[1] = data[i][1];
			p = data[i][2];
			backPropagation(input, association, output, weigths_1, weigths_2, biases_1, biases_2, p, grad_1, grad_2, grad_b1, grad_b2);
			for (int q = 0; q < 2; q++)
			{
				av_grad_2[q] += grad_2[q];

				av_grad_b1[q] += grad_b1[q];

				for (int p = 0; p < 2; p++)
				{
					av_grad_1[q][p] += grad_1[q][p];
				}
			}

			av_grad_b2 += grad_b2;
		}

		for (int q = 0; q < 2; q++)
		{
			av_grad_2[q] /= 4;

			av_grad_b1[q] /= 4;

			for (int p = 0; p < 2; p++)
			{
				av_grad_1[q][p] /= 4;
			}
		}

		av_grad_b2 += grad_b2;

		learn(weigths_1, av_grad_1, weigths_2, av_grad_2, biases_1, av_grad_b1, biases_2, av_grad_b2);
	}

	cout << endl << "Результаты обучения: ";

	printConfig(weigths_1, weigths_2, biases_1, biases_2);

	int pick;

	do
	{
		cout << "Решение задачи XOR" << endl;
		cout << "Выберете режим работы: " << endl;
		cout << "1 - Прямое распространение" << endl;
		cout << "2 - Обратное распространение" << endl;
		cout << "// Любое другое число для выхода" << endl;

		cout << "Выбор: ";
		cin >> pick;

		inputData(input);

		switch (pick)
		{
		case 1:
			forwordPropagation(input, association, output, weigths_1, weigths_2, biases_1, biases_2);
			printAnswer(output);
			break;
		case 2:
			cout << "Введите эталон: ";
			cin >> p;
			backPropagation(input, association, output, weigths_1, weigths_2, biases_1, biases_2, p, grad_1, grad_2, grad_b1, grad_b2);
			learn(weigths_1, grad_1, weigths_2, grad_2, biases_1, grad_b1, biases_2, grad_b2);
			printConfig(weigths_1, weigths_2, biases_1, biases_2);
			break;
		default:
			return;
			break;
		}
	} while (pick == 1 || pick == 2);
}