// Practica 2 - Seccion 3.2: Analisis de memoria
// NOTA: la clase aqui es una version minima con SOLO los atributos pedidos
// (weights y bias). Si la clase final agrega mas atributos (ej. medias/desv.
// para el escalado), sizeof(LinearRegression) cambia: volver a ejecutar.
#include <iostream>
#include <vector>
using namespace std;

class LinearRegression {
private:
    vector<double> weights;  // 3 punteros internos (inicio, fin, capacidad) = 24 bytes
    double bias;             // 8 bytes
public:
    LinearRegression(int nFeatures = 0) : weights(nFeatures, 0.0), bias(0.0) {}
    const double* weightsData() const { return weights.data(); }  // para ver el buffer en heap
};

// 1) Memoria GLOBAL (segmento de datos)
LinearRegression globalModel(3);

void show(const string& name, const void* addr, size_t size) {
    cout << name << "\n   direccion: " << addr << "\n   sizeof:    " << size << " bytes\n";
}

int main() {
    cout << "===== Objetos individuales =====\n";

    // 2) Memoria LOCAL (stack)
    LinearRegression localModel(3);

    // 3) Memoria DINAMICA (heap)
    LinearRegression* dynModel = new LinearRegression(3);

    show("Global  (globalModel)", &globalModel, sizeof(globalModel));
    show("Local   (localModel)", &localModel, sizeof(localModel));
    show("Dinamico: el puntero dynModel (vive en stack)", &dynModel, sizeof(dynModel));
    show("Dinamico: el objeto apuntado (vive en heap)", dynModel, sizeof(*dynModel));

    cout << "\n--- Buffer interno de weights (siempre en heap) ---\n";
    cout << "globalModel.weights -> " << globalModel.weightsData() << "\n";
    cout << "localModel.weights  -> " << localModel.weightsData() << "\n";
    cout << "dynModel->weights   -> " << dynModel->weightsData() << "\n";
    cout << "(3 doubles = " << 3 * sizeof(double) << " bytes, NO cuentan en sizeof del objeto)\n";

    cout << "\n===== Vectores de LinearRegression en memoria dinamica =====\n";

    // Opcion A: arreglo dinamico estilo C
    const int N = 5;
    LinearRegression* arr = new LinearRegression[N];
    show("Arreglo new LinearRegression[5]", arr, N * sizeof(LinearRegression));
    cout << "   sizeof(elemento) = " << sizeof(LinearRegression)
         << " -> total = " << N << " x " << sizeof(LinearRegression)
         << " = " << N * sizeof(LinearRegression) << " bytes\n";
    cout << "   direccion elem[0]=" << &arr[0] << "  elem[1]=" << &arr[1] << "\n";

    // Opcion B: std::vector alojado en el heap
    vector<LinearRegression>* vec = new vector<LinearRegression>(N, LinearRegression(3));
    show("new vector<LinearRegression>(5)  (objeto vector)", vec, sizeof(*vec));
    cout << "   buffer de elementos en heap: " << vec->data() << "\n";
    cout << "   bytes en el buffer = capacity x sizeof(elemento) = "
         << vec->capacity() << " x " << sizeof(LinearRegression)
         << " = " << vec->capacity() * sizeof(LinearRegression) << " bytes\n";
    cout << "   sizeof(vector) NO depende de la cantidad de elementos: "
         << sizeof(*vec) << " bytes\n";

    // Liberar memoria
    delete dynModel;
    delete[] arr;
    delete vec;
    return 0;
}
