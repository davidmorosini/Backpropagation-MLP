#include "globais.h"
#include "MLP.cpp"


int main(int argc, char * argv[]){
	srand((unsigned)time(0));

	float input_inicial[5] = {0.6666, 0.8233, 0.99902, 0.022, 0.4};
	MLP rede_neural = inicializa_MLP(5, 6, 10, 4, input_inicial);

	//inicializa_MLP(int qtd_inputs, int qtd_saidas, int qtd_camadas, int qtd_neuronios_camada, float * input_inicial){

	int saida_esperada[6] = {1, 0, 1, 0, 0, 0};


	propaga_sinal(rede_neural, saida_esperada);

	//propaga_erro(rede_neural);

	imprime(rede_neural);

	exibe_codigo_gerado(rede_neural, saida_esperada);
	
}