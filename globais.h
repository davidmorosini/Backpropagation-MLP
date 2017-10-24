#ifndef GLOBAIS_H
#define GLOBAIS_H

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;

#define RANGE_PESO 100 	//sera gerado um numero entre 0 e RANGE_PESO 
						//e esse valor sera dividido por RANGE_PESO
						//de modo a gerar um valor entre 0.0 e 1.0

#define BIAS -0.999999

//////////////////////////////  PERCEPTRON

typedef struct perceptron * neuronio;
typedef neuronio * camada;				//somente para ficar mais legivel
typedef camada * blackbox;				//rede neural 


typedef struct perceptron{
	//conexoes que cada neuronio faz, com os anteriores e os posteriores
	int qtd_conexoes_back, qtd_conexoes_front;
	camada back, front;

	float sinal;
	float erro;

	//pesos e inputs de cada neuronio, provindos das ligacoes anteriores
	float * pesos;  	//qtd_conexoes_back + 1, lugar reservado para o BIAS
	float * inputs;

}perceptron;



typedef struct MLP{
	int qtd_inputs,
		qtd_saidas,
		qtd_neuronios_rede,
		qtd_camadas_internas;

	camada entrada; 	//"camada de entrada"
	camada saida; 	//"camada de saida" 

	float erro_medio_saida;

	blackbox rede;   	
}MLP;



//variaveis necessárias para o processo de aprendizagem
float taxa_aprendizagem = 0.2;
float aprendizado = 0.0;

float erro = 0.1;  //10% de erro



#endif