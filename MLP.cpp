#include "globais.h"


//funcao inicializa neuronio
neuronio init_neuronio(int qtd_conexoes_back, int qtd_conexoes_front){
	neuronio novo = (neuronio)malloc(sizeof(perceptron));
	if(novo != NULL){

		novo -> sinal = 0;
		novo -> erro = 0;
		//contando o BIAS
		novo -> qtd_conexoes_back = qtd_conexoes_back + 1;
		novo -> qtd_conexoes_front = qtd_conexoes_front;
		novo -> back = NULL;
		novo -> front = NULL;

		float * pesos = (float *)malloc((qtd_conexoes_back) * sizeof(float));
		float * inputs = (float *)malloc((qtd_conexoes_back) * sizeof(float));

		int aux = rand() % (RANGE_PESO + 1);
		pesos[0] = (float)aux/(float)RANGE_PESO;
		inputs[0] = BIAS; //fator inicial de inibição do neuronio

		//posicao 0 é o BIAS
		for(int i = 1; i <= qtd_conexoes_back; i++){
			int aux = rand() % (RANGE_PESO + 1);
			pesos[i] = (float)aux/(float)RANGE_PESO;
			inputs[i] = 0;
		}

		novo -> pesos = pesos;
		novo -> inputs = inputs;
	}

	return novo;
}

camada init_camada(int qtd_neuronios, int qtd_conexoes_back, int qtd_conexoes_front){
	camada nova = (camada)malloc(qtd_neuronios * sizeof(neuronio));
	
	if(nova != NULL){
		for(int i = 0; i < qtd_neuronios; i++){
			nova[i] = init_neuronio(qtd_conexoes_back, qtd_conexoes_front);
		}
	}

	return nova;
}

//blackbox são as camadas internas de neuronios
blackbox init_blackbox(int qtd_neuronios, int qtd_camadas){
	blackbox rede = (blackbox)malloc(qtd_camadas * sizeof(camada));
	
	if(rede != NULL){
		for(int i = 0; i < qtd_camadas; i++){
			rede[i] = init_camada(qtd_neuronios, qtd_neuronios, qtd_neuronios);
		}
		//realizando as ligacoes ente as camadas internas
		for(int i = 0; i < qtd_camadas; i++){
			if(i > 0){// todos estes possuem um sucessor
				for(int j = 0; j < qtd_neuronios; j++){
					((rede[i])[j]) -> back = rede[i - 1];
				}
			}

			if( i < (qtd_camadas - 1)){//todos estes possuem um antecessor
				for(int j = 0; j < qtd_neuronios; j++){
					((rede[i])[j]) -> front = rede[i + 1];
				}
			}
		}
	}
	return rede;
}

float funcao_ativacao(float num){
	return (1 / (1 + pow(M_E, -num)));
}

//propaga o sinal desde a camada de entrada até a camada de saida
//passando pela HIDDEN LAYER
void propaga_sinal(MLP rede, int * saida_esperada){
	float sum = 0, sinal = 0;

	//aqui vem o tratamento especial para a primeira camada,a  de entrada:
	for(int i = 0; i < rede.qtd_inputs; i++){
		sum = 0;
		for(int k = 0; k < (rede.entrada)[i] -> qtd_conexoes_back; k++){
			sum += ((rede.entrada)[i] -> inputs)[k] * ((rede.entrada)[i] -> pesos)[k];
		}
		sinal = funcao_ativacao(sum);
		((rede.entrada)[i] -> sinal) = sinal;

		for(int j = 0; j < rede.qtd_neuronios_rede; j++){
			(((rede.rede)[0])[j] -> inputs)[i + 1] = sinal;
		}
	}

	//rede neural interna
	//(rede.qtd_camadas_internas - 1) pq abaixo deve vir o algoritmo para tratar a camada de saida
	for(int i = 0; i < rede.qtd_camadas_internas; i++){
		sum = 0;
		for(int j = 0; j < rede.qtd_neuronios_rede; j++){
			//qtd_conexoes_back ja inclui o BIAS
			for(int k = 0; k < (((rede.rede)[i])[j] -> qtd_conexoes_back); k++){
				sum += (((rede.rede)[i])[j] -> inputs)[k] * (((rede.rede)[i])[j] -> pesos)[k];
			}
			sinal = funcao_ativacao(sum);

			(((rede.rede)[i])[j] -> sinal) = sinal;

			//cout << "sinal " << sinal << endl;

			//replicando o sinal para a frente da rede
			if(i + 1 < rede.qtd_neuronios_rede){
				for(int k = 0; k < (((rede.rede)[i])[j] -> qtd_conexoes_front); k++){
					((((rede.rede)[i])[j] -> front)[k] -> inputs)[j+1] = sinal;
				}
			}else{
				//passando para a camada de saida
				for(int k = 0; k < rede.qtd_saidas; k++){
					((rede.saida)[k] -> inputs)[j+1] = sinal;
				}
			}
			

		}	
	}


	//camada de saida dos dados
	for(int i = 0; i < rede.qtd_saidas; i++){
		sum = 0;
		for(int k = 0; k < (rede.saida)[i] -> qtd_conexoes_back; k++){
			sum += ((rede.saida)[i] -> inputs)[k] * ((rede.saida)[i] -> pesos)[k];
		}
		sinal = funcao_ativacao(sum);
		(rede.saida)[i] -> sinal = sinal;
		(rede.saida)[i] -> erro = saida_esperada[i] - sinal;
	}
}
///////////////////////////////////////////////////////////////  AQUI
void propaga_erro(MLP rede){
	//int * resp -> possui a resposta em binario
	//a retropropagação do erro comeca a partir da ultima camada da HIDDEN LAYER
	//para a primeira camada HIDDEN LAYER
	for(int i = (rede.qtd_camadas_internas - 1); i >= 0; i++){
		//ireações para cada neuronio de cada camada da HIDDEN LAYER
		for(int j = 0; j < rede.qtd_neuronios_rede; j++){
			float erro_sum = 0;
			//para cada neuronio, fazer o somatorio do erro_sucessor * peso_saida
			for(int k = 0; k < ((((rede.rede)[i])[j]) -> qtd_conexoes_front); k++){
				//somatorio dos erros * pesos de entrada posteriores
				float erro_prox = (((rede.rede)[i])[j] -> front)[k] -> erro;
				//cout << "passou" << endl;
				//pesos[j]  -> referente ao peso deste neuronio no neuronio seguinte
				float peso_prox = ((((rede.rede)[i])[j] -> front)[k] -> pesos)[j];
				erro_sum +=  erro_prox * peso_prox;
			}
			((((rede.rede)[i])[j]) -> erro) = erro_sum;
		}
	}


	

}

///////////////////////////////////////////////////////////////  AQUI

void imprime(MLP rede){
	cout << endl << "IMPRIMINDO A MLP" << endl;

	cout << "QTD_NEURONIOS_INPUTS: " << rede.qtd_inputs << endl;
	cout << "QTD_NEURONIOS_SAIDAS: " << rede.qtd_saidas << endl;
	cout << "QTD_NEURONIOS_INTERNOS: " << rede.qtd_neuronios_rede << endl;
	cout << "QTD_CAMADAS_INTERNAS: " << rede.qtd_camadas_internas << endl;

	cout << endl << endl << "#################  CAMADA DE ENTRADA   #################";
	for(int j = 0; j < rede.qtd_inputs; j++){
		cout << endl << endl << "NEURONIO " << j << endl;
		//somente o BIAS e uma entrada
		for(int k = 0; k < 2; k++){
			cout << "\t" << "INPUT("<<k<<"): " << ((rede.entrada)[j] -> inputs)[k];
			cout << "\t" << "PESO("<<k<<"): " << ((rede.entrada)[j] -> pesos)[k] << endl; 
		}
		cout << endl;
	}

	cout << endl << endl << "#################  HIDDEN LAYER   #################";
	for(int i = 0; i < rede.qtd_camadas_internas; i++){
		cout << endl << endl << "CAMADA " << i << endl;
		for(int j = 0; j < rede.qtd_neuronios_rede; j++){
			cout << "NEURONIO " << j << endl;
			for(int k = 0; k < rede.qtd_neuronios_rede + 1; k++){
				cout << "\t" << "INPUT("<<k<<"): " << ((((rede.rede)[i])[j]) -> inputs)[k];
				cout << "\t" << "PESO("<<k<<"): " << ((((rede.rede)[i])[j]) -> pesos)[k] << endl; 
			}
			cout << endl;
		}
	}

	cout << endl << endl << "#################  CAMADA DE SAIDA   #################";
	for(int j = 0; j < rede.qtd_saidas; j++){
		cout << endl << endl << "NEURONIO " << j << endl;
		//somente o BIAS e uma entrada
		for(int k = 0; k < (rede.saida)[j] -> qtd_conexoes_back; k++){
			cout << "\t" << "INPUT("<<k<<"): " << ((rede.saida)[j] -> inputs)[k];
			cout << "\t" << "PESO("<<k<<"): " << ((rede.saida)[j] -> pesos)[k] << endl; 
		}
		cout << endl;
	}

	cout << endl << endl;
}

MLP inicializa_MLP(int qtd_inputs, int qtd_saidas, int qtd_camadas, int qtd_neuronios_camada, float * input_inicial){
	MLP new_mlp;
	new_mlp.qtd_inputs = qtd_inputs;
	new_mlp.qtd_saidas = qtd_saidas;
	new_mlp.qtd_neuronios_rede = qtd_neuronios_camada;
	new_mlp.qtd_camadas_internas = qtd_camadas;

	//cada neuronio da entrada processa apenas um input e o BIAS
	//o BIAS esta iniciado internamente na funcao init_camada
	new_mlp.entrada = init_camada(qtd_inputs, 1, qtd_neuronios_camada);
	if(input_inicial != NULL){
		for(int i = 0; i < qtd_inputs; i++){
			//posicao 0 esta armazenado o BIAS
			((new_mlp.entrada)[i] -> inputs)[1] = input_inicial[i];
		}
	}

	new_mlp.saida = init_camada(qtd_saidas, qtd_neuronios_camada, 0);

	new_mlp.rede = init_blackbox(qtd_neuronios_camada, qtd_camadas);

	return new_mlp;
}

void exibe_codigo_gerado(MLP rede, int * saida_esperada){
	float erro = 0;
	cout << endl << endl << "SAIDA EM BINARIO DA REDE: " << endl;
	for(int j = 0; j < rede.qtd_saidas; j++){
		int num = 0;
		if((rede.saida)[j] -> sinal > 0.5){
			num = 1;
		}
		if(num != saida_esperada[j]){
			erro++;
		}
		cout << "calculado: " << num << ", esperado: " << saida_esperada[j] << endl;
	}

	erro /= (rede.qtd_saidas);

	cout << endl << "Taxa de acertos " << (1 - erro) * 100 << "%" << endl; 
	

	cout << endl << endl;
}

void treinamento_rede(MLP rede, float * entrada, int * saida_esperada){

}