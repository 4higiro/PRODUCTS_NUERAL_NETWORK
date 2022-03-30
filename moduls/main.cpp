// ���������� �����
#include <iostream>
#include <fstream>

#include "network.h"

// ������������ �����
using namespace std;

// ���� ������ 
void inputData(float* input, int n)
{
	for (int i = 0; i < n; i++)
	{
		cout << "���� " << i << ": ";
		cin >> input[i];
	}

	cout << endl;
}

// ���� �������� ��� ��������
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

// ����� ����� � ���������
void main()
{
	// ��������� �������� �����
	setlocale(LC_ALL, "Ru");

	// ������ ���������������
	cout << "��������� ���������� ������ XOR" << endl;

	int n = 3;
	int layers[3] = { 2, 2, 1 };
	float alpha = 1;
	float betta = 0.5;

	cout << "������������ ����: " << "\t";

	for (int i = 0; i < n; i++)
	{
		cout << layers[i] << "\t";
	}

	cout << endl << "�������� ��������: " << alpha << "  ������: " << betta << endl << endl;

	// ���������� �������� ��� �������� �� �����
	ifstream file;
	file.open("data.txt");

	if (!file.is_open())
	{
		cout << "���� �� ������" << endl;
		return;
	}

	int example_count;
	file >> example_count;

	float** examples = new float* [example_count];
	float** predictions = new float* [example_count];

	for (int i = 0; i < example_count; i++)
	{
		examples[i] = new float[layers[0]];
		predictions[i] = new float[layers[n - 1]];
	}

	for (int i = 0; i < example_count; i++)
	{
		for (int j = 0; j < layers[0]; j++)
		{
			file >> examples[i][j];
		}
	}

	for (int i = 0; i < example_count; i++)
	{
		for (int j = 0; j < layers[n - 1]; j++)
		{
			file >> predictions[i][j];
		}
	}

	file.close();

	// ������������� ���������
	nueral_network net;
	net.init(layers, n, alpha, betta);

	int pick = 0;
	// �������� ���� ���������
	do
	{
		// ����
		cout << "�������� ������ ����� ����:" << endl;
		cout << "1 - ������ ���������������" << endl;
		cout << "2 - �������� ���������������" << endl;
		cout << "3 - �������� �� ������ �� �����" << endl;
		cout << "// ������� ��� ������ ��� ���������� ������" << endl;

		cout << "�����: ";
		cin >> pick;

		float* input = new float[layers[0]];
		float* output = new float[layers[n - 1]];
		float result = 0;
		int answer = 0;

		switch (pick)
		{
		case 1:
			inputData(input, layers[0]);
			answer = net.forwordPropagetion(result, input);
			cout << "����� ���������: " << answer << " (" << result << ")" << endl;
			break;
		case 2:
			inputLearnData(input, output, layers[0], layers[n - 1]);
			net.printWeights();
			net.backPropagetion(input, output);
			net.learn();
			net.printWeights();
			cout << endl;
			break;
		case 3:
			net.printWeights();
			for (int k = 0; k < 10000; k++)
			{
				for (int i = 0; i < example_count; i++)
				{
					net.backPropagetion(examples[i], predictions[i]);
					net.learn();
				}
			}
			net.printWeights();
			break;
		default:
			return;
		}
	} while (pick == 1 || pick == 2 || pick == 3);
	
	system("pause");
}