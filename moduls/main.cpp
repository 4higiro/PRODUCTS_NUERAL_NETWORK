#include <iostream>
#include <fstream>

#include "network.h"

using namespace std;

void inputData(float* input, int n)
{
	for (int i = 0; i < n; i++)
	{
		cout << "���� " << i << ": ";
		cin >> input[i];
	}

	cout << endl;
}

void inputLearnData(float* input, float* p, int n, int m)
{
	for (int i = 0; i < n; i++)
	{
		cout << "���� " << i << ": ";
		cin >> input[i];
	}

	for (int i = 0; i < m; i++)
	{
		cout << "������ " << i << ": ";
		cin >> p[i];
	}

	cout << endl;
}

void main()
{
	setlocale(LC_ALL, "Ru");

	cout << "��������� ���������� ������ XOR" << endl;

	int n = 3;
	int layers[3] = { 2, 2, 1 };

	cout << "������������ ����: " << "\t";

	for (int i = 0; i < n; i++)
	{
		cout << layers[i] << "\t";
	}

	cout << endl;

	ifstream file;
	file.open("resources/data.txt");

	if (!file.is_open())
	{
		cout << "���� �� ������" << endl;
		return;
	}

	int count;

	file >> count;

	float** examples = new float* [count];

	for (int i = 0; i < count; i++)
	{
		examples[i] = new float[layers[0]];

		for (int j = 0; j < layers[0]; j++)
		{
			file >> examples[i][j];
		}
	}

	float** predictions = new float* [count];

	for (int i = 0; i < count; i++)
	{
		predictions[i] = new float[layers[n - 1]];

		for (int j = 0; j < layers[n - 1]; j++)
		{
			file >> predictions[i][j];
		}
	}

	file.close();

	nueral_network net;
	net.init(layers, n);

	int pick = 0;

	do
	{
		cout << "�������� ������ ����� ����:" << endl;
		cout << "1 - ������ ���������������" << endl;
		cout << "2 - �������� ���������������" << endl;
		cout << "// ������� ��� ������ ��� ���������� ������" << endl;

		cout << "�����: ";
		cin >> pick;

		float* input = new float[layers[0]];
		float result = 0;
		int answer = 0;

		switch (pick)
		{
		case 1:
			inputData(input, layers[0]);
			result = 0;
			answer = net.forwordPropagetion(result, input);
			cout << "����� ���������: " << answer << " (" << result << ")" << endl;
			net.printNuerals();
			cout << endl;
			break;
		case 2:
			cout << "��������e �������:" << endl;
			for (int i = 0; i < count; i++)
			{
				for (int j = 0; j < layers[0]; j++)
				{
					cout << examples[i][j] << "  ";
				}

				cout << " -->   ";

				for (int j = 0; j < layers[n - 1]; j++)
				{
					cout << predictions[i][j] << "  ";
				}

				cout << endl;
			}
			net.printWeights();
			for (int i = 0; i < count; i++)
			{
				net.backPropagetion(examples[i], predictions[i]);
				net.printNuerals();
			}
			net.learn();
			net.printWeights();
			break;
		default:
			return;
			break;
		}


	} while (pick == 1 || pick == 2);
	
	system("pause");
}