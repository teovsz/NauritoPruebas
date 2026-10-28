import numpy as np

ACCIONES = ("comer", "dormir", "jugar")


def politica(X):
    # Igual que tu main.py: gana la necesidad más alta (cansancio = 1 - energía)
    return np.argmax(np.column_stack((X[:, 0], 1 - X[:, 1], X[:, 2])), axis=1)


def relu(v):
    return np.maximum(0, v)


def softmax(v):
    e = np.exp(v - np.max(v, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def predecir(X, red):
    W1, b1, W2, b2 = red
    return softmax(relu(X @ W1 + b1) @ W2 + b2)


def perdida(p, y):
    return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-10))


def entrenar(ocultas=6, epocas=1200, lr=0.1):
    np.random.seed(42)
    X = np.random.rand(3000, 3)
    y = politica(X)
    W1 = np.random.randn(3, ocultas) * 0.5
    b1 = np.zeros((1, ocultas))
    W2 = np.random.randn(ocultas, 3) * 0.5
    b2 = np.zeros((1, 3))
    n = len(X)
    historial = []
    for _ in range(epocas):
        z1 = X @ W1 + b1
        h = relu(z1)
        p = softmax(h @ W2 + b2)
        historial.append(perdida(p, y))
        d2 = p.copy()
        d2[np.arange(n), y] -= 1
        d2 /= n
        gW2 = h.T @ d2
        gb2 = d2.sum(axis=0, keepdims=True)
        d1 = (d2 @ W2.T) * (z1 > 0)
        gW1 = X.T @ d1
        gb1 = d1.sum(axis=0, keepdims=True)
        W1 -= lr * gW1
        b1 -= lr * gb1
        W2 -= lr * gW2
        b2 -= lr * gb2
    return (W1, b1, W2, b2), historial


def evaluar(red):
    # Conjunto de prueba fijo, para comparar justamente entre experimentos
    Xt = np.random.RandomState(123).rand(1000, 3)
    yt = politica(Xt)
    p = predecir(Xt, red)
    return perdida(p, yt), np.mean(p.argmax(axis=1) == yt), Xt, yt, p


def decidir(red, hambre, energia, aburr):
    p = predecir(np.array([[hambre, energia, aburr]]), red)[0]
    return ACCIONES[int(p.argmax())], np.round(p, 3)


if __name__ == "__main__":
    red, hist = entrenar()
    l, a, Xt, yt, p = evaluar(red)
    print(f"BASE (6 neuronas, 1200 épocas, lr=0.1): loss={l:.3f} acc={a:.2%}")

    print("\nExp 1: hambre=0.5 y aburrimiento=0.5 fijos, varía la energía")
    for e in [0.9, 0.7, 0.55, 0.5, 0.45, 0.3, 0.1]:
        print(f"  energía={e:.2f} (cansancio={1 - e:.2f}):", decidir(red, 0.5, e, 0.5))

    print("\nExp 2: prioridades casi empatadas vs. claras")
    print("  hambre 0.80 / cansancio 0.80:", decidir(red, 0.80, 0.20, 0.10))
    print("  hambre 0.80 / cansancio 0.81:", decidir(red, 0.80, 0.19, 0.10))
    print("  hambre 0.90 / cansancio 0.20:", decidir(red, 0.90, 0.80, 0.10))
    print("  triple empate 0.5/0.5/0.5   :", decidir(red, 0.50, 0.50, 0.50))

    print("\nExp 3: neuronas ocultas")
    for H in [3, 6, 10, 20]:
        r, _ = entrenar(ocultas=H)
        l, a, *_ = evaluar(r)
        print(f"  H={H:2d}: loss={l:.3f}  acc={a:.2%}")

    print("\nExp 4a: épocas (lr=0.1)")
    for ep in [100, 400, 1200, 5000]:
        r, h = entrenar(epocas=ep)
        l, a, *_ = evaluar(r)
        print(f"  épocas={ep:4d}: loss={l:.3f}  acc={a:.2%}")

    print("\nExp 4b: tasa de aprendizaje (1200 épocas)")
    for lr in [0.001, 0.01, 0.1, 1, 10]:
        r, h = entrenar(lr=lr)
        l, a, *_ = evaluar(r)
        subidas = int(np.sum(np.diff(h) > 1e-9))
        print(f"  lr={lr:6}: loss={l:.3f}  acc={a:.2%}  (veces que la pérdida subió: {subidas})")

    print("\nPara la pregunta 4: precisión según dificultad")
    top2 = np.sort(np.column_stack((Xt[:, 0], 1 - Xt[:, 1], Xt[:, 2])), axis=1)
    brecha = top2[:, 2] - top2[:, 1]
    acierto = p.argmax(axis=1) == yt
    print(f"  casos claros (brecha >= 0.05):  {acierto[brecha >= 0.05].mean():.2%}")
    print(f"  casi empates (brecha < 0.05):   {acierto[brecha < 0.05].mean():.2%}"
          f"  ({np.sum(brecha < 0.05)} casos)")
    for c, nombre in enumerate(ACCIONES):
        print(f"  precisión en '{nombre}': {acierto[yt == c].mean():.2%}")