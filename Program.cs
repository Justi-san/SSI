using System.Globalization;
using System.Text;

namespace Grupowanie;

// ---------------------------------------------------------------------------
//  Ćwiczenie: grupowanie (k-średnich i Fuzzy c-Means) na zbiorze "spiralka"
//  Uruchomienie:  dotnet run
//  Wyniki:  konsola + folder "wyniki" (raport.txt oraz wykresy SVG)
// ---------------------------------------------------------------------------

delegate double Odleglosc(double[] a, double[] b);

record KMeansStan(int Iteracja, double[][] Srodki, int[] Przydzial);
record FcmStan(int Iteracja, double[][] Srodki, double[,] U);

static class Program
{
    const int Ziarno = 123;               // stałe ziarno -> powtarzalne wyniki (zmień, by zobaczyć inne losowanie)
    const double PrógFcm = 0.6;           // próg przynależności w raporcie FCM
    const string Folder = "wyniki";

    static readonly StringBuilder Raport = new();

    // ------------------------- MIARY ODLEGŁOŚCI -----------------------------
    static double Euklides(double[] a, double[] b)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++) { double d = a[i] - b[i]; s += d * d; }
        return Math.Sqrt(s);
    }

    // tylko atrybut x1, x2 ignorowany
    static double TylkoX1(double[] a, double[] b) => Math.Abs(a[0] - b[0]);

    // (pozostałe miary z instrukcji - niewymagane w raportach, ale dołączone)
    static double Manhattan(double[] a, double[] b)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++) s += Math.Abs(a[i] - b[i]);
        return s;
    }

    static double Czebyszew(double[] a, double[] b)
    {
        double m = 0;
        for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]));
        return m;
    }

    static double Minkowski(double[] a, double[] b, double mink)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++) s += Math.Pow(Math.Abs(a[i] - b[i]), mink);
        return Math.Pow(s, 1.0 / mink);
    }

    // ------------------------------ K-ŚREDNICH ------------------------------
    static List<KMeansStan> KMeans(double[][] X, int m, int iters, Odleglosc dist,
                                   Random rng, ISet<int> zapisz)
    {
        int M = X.Length, n = X[0].Length;
        var wyniki = new List<KMeansStan>();

        // 1. losowo m różnych próbek jako środki grup
        var idx = Enumerable.Range(0, M).OrderBy(_ => rng.Next()).Take(m).ToArray();
        var V = idx.Select(i => (double[])X[i].Clone()).ToArray();
        var u = new int[M];

        // 2. pętla iteracji
        for (int it = 1; it <= iters; it++)
        {
            // 2.1 przydział próbek do najbliższego środka
            for (int s = 0; s < M; s++)
            {
                int best = 0;
                double bd = dist(X[s], V[0]);
                for (int j = 1; j < m; j++)
                {
                    double d = dist(X[s], V[j]);
                    if (d < bd) { bd = d; best = j; }
                }
                u[s] = best;
            }

            // 2.2 wyznaczenie nowych środków
            for (int j = 0; j < m; j++)
            {
                int cnt = 0;
                var sum = new double[n];
                for (int s = 0; s < M; s++)
                {
                    if (u[s] != j) continue;
                    cnt++;
                    for (int i = 0; i < n; i++) sum[i] += X[s][i];
                }
                if (cnt == 0) continue;                       // pusta grupa - środek bez zmian
                for (int i = 0; i < n; i++) V[j][i] = sum[i] / cnt;
            }

            if (zapisz.Contains(it))
                wyniki.Add(new KMeansStan(it, V.Select(v => (double[])v.Clone()).ToArray(), (int[])u.Clone()));
        }
        return wyniki;
    }

    // ------------------------- FUZZY C-MEANS --------------------------------
    static List<FcmStan> Fcm(double[][] X, int m, int iters, double fcmM,
                             Odleglosc dist, Random rng, ISet<int> zapisz)
    {
        int M = X.Length, n = X[0].Length;
        var D = new double[m, M];
        var U = new double[m, M];
        var V = new double[m][];
        for (int j = 0; j < m; j++) V[j] = new double[n];
        var wyniki = new List<FcmStan>();

        // 1.3 D - losowe wartości z przedziału (0.1; 1.1)
        for (int j = 0; j < m; j++)
            for (int s = 0; s < M; s++)
                D[j, s] = 0.1 + rng.NextDouble();

        // 1.4 + 1.5 początkowe U i V
        ObliczU(D, U, m, M, fcmM);
        ObliczV(X, U, V, m, M, n, fcmM);

        // 2. główna pętla
        for (int it = 1; it <= iters; it++)
        {
            // 2.1 odległości (kwadrat) próbka - środek
            for (int j = 0; j < m; j++)
                for (int s = 0; s < M; s++)
                {
                    double d = dist(X[s], V[j]);
                    D[j, s] = d * d;
                    if (D[j, s] < 1e-5) D[j, s] = 1e-5;       // 2.2 zabezpieczenie
                }

            // 2.3 przynależności
            ObliczU(D, U, m, M, fcmM);

            // 2.4 kontrola wartości nieoznaczonych
            for (int j = 0; j < m; j++)
                for (int s = 0; s < M; s++)
                    if (double.IsNaN(U[j, s]) || double.IsInfinity(U[j, s]))
                    {
                        Console.WriteLine($"OSTRZEŻENIE: U zawiera wartość nieoznaczoną (iteracja {it}). Przerywam.");
                        return wyniki;
                    }

            // 2.5 nowe środki
            ObliczV(X, U, V, m, M, n, fcmM);

            if (zapisz.Contains(it))
                wyniki.Add(new FcmStan(it, V.Select(v => (double[])v.Clone()).ToArray(), (double[,])U.Clone()));
        }
        return wyniki;
    }

    static void ObliczU(double[,] D, double[,] U, int m, int M, double fcmM)
    {
        double p = 1.0 / (1.0 - fcmM);
        for (int s = 0; s < M; s++)
        {
            double suma = 0;
            for (int j = 0; j < m; j++) suma += Math.Pow(D[j, s], p);
            for (int j = 0; j < m; j++) U[j, s] = Math.Pow(D[j, s], p) / suma;
        }
    }

    static void ObliczV(double[][] X, double[,] U, double[][] V, int m, int M, int n, double fcmM)
    {
        for (int j = 0; j < m; j++)
            for (int i = 0; i < n; i++)
            {
                double licznik = 0, mianownik = 0;
                for (int s = 0; s < M; s++)
                {
                    double w = Math.Pow(U[j, s], fcmM);
                    licznik += w * X[s][i];
                    mianownik += w;
                }
                V[j][i] = licznik / mianownik;
            }
    }

    // ------------------------------ RAPORTY ---------------------------------
    static void Pisz(string tekst = "")
    {
        Console.WriteLine(tekst);
        Raport.AppendLine(tekst);
    }

    static string NazwaPliku(string nazwa) =>
        System.Text.RegularExpressions.Regex.Replace(nazwa, "[^A-Za-z0-9]+", "_").Trim('_');

    static string F(double x) => x.ToString("F4", CultureInfo.InvariantCulture);

    static void OpisGrupy(int nr, double[] srodek, List<double[]> pkt)
    {
        string poz = $"({F(srodek[0])}; {F(srodek[1])})";
        if (pkt.Count == 0)
        {
            Pisz($"  Grupa {nr}: środek = {poz}, liczba próbek = 0 (grupa pusta)");
            return;
        }
        Pisz($"  Grupa {nr}: środek = {poz}, liczba próbek = {pkt.Count}, " +
             $"x1 in [{F(pkt.Min(p => p[0]))}; {F(pkt.Max(p => p[0]))}], " +
             $"x2 in [{F(pkt.Min(p => p[1]))}; {F(pkt.Max(p => p[1]))}]");
    }

    static void RaportKMeans(string nazwa, double[][] X, int m, KMeansStan st)
    {
        Pisz($"--- {nazwa}: stan po {st.Iteracja} iteracjach ---");
        var grupy = new List<List<double[]>>();
        for (int j = 0; j < m; j++)
        {
            var pkt = new List<double[]>();
            for (int s = 0; s < X.Length; s++) if (st.Przydzial[s] == j) pkt.Add(X[s]);
            grupy.Add(pkt);
            OpisGrupy(j + 1, st.Srodki[j], pkt);
        }
        string plik = $"{NazwaPliku(nazwa)}_iter{st.Iteracja}.svg";
        Svg.Zapisz(Path.Combine(Folder, plik), $"{nazwa}, iteracji: {st.Iteracja}",
                   grupy.Select((g, j) => ($"Grupa {j + 1} ({g.Count})", g)).ToList(), st.Srodki);
        Pisz($"  Wykres: {Path.Combine(Folder, plik)}");
        Pisz();
    }

    static void RaportFcm(string nazwa, double[][] X, int m, FcmStan st)
    {
        Pisz($"--- {nazwa}: stan po {st.Iteracja} iteracjach (próg przynależności > {PrógFcm.ToString(CultureInfo.InvariantCulture)}) ---");
        var grupy = new List<List<double[]>>();
        for (int j = 0; j < m; j++)
        {
            var pkt = new List<double[]>();
            for (int s = 0; s < X.Length; s++) if (st.U[j, s] > PrógFcm) pkt.Add(X[s]);
            grupy.Add(pkt);
            OpisGrupy(j + 1, st.Srodki[j], pkt);
        }
        var niezdec = new List<double[]>();
        for (int s = 0; s < X.Length; s++)
        {
            bool ok = false;
            for (int j = 0; j < m; j++) if (st.U[j, s] > PrógFcm) ok = true;
            if (!ok) niezdec.Add(X[s]);
        }
        Pisz($"  Próbki bez grupy o przynależności > {PrógFcm.ToString(CultureInfo.InvariantCulture)}: {niezdec.Count}");

        var serie = grupy.Select((g, j) => ($"Grupa {j + 1}, U>0.6 ({g.Count})", g)).ToList();
        serie.Add(($"U<=0.6 ({niezdec.Count})", niezdec));
        string plik = $"{NazwaPliku(nazwa)}_iter{st.Iteracja}.svg";
        Svg.Zapisz(Path.Combine(Folder, plik), $"{nazwa}, iteracji: {st.Iteracja}", serie, st.Srodki);
        Pisz($"  Wykres: {Path.Combine(Folder, plik)}");
        Pisz();
    }

    // -------------------------------- MAIN ----------------------------------
    static void Main()
    {
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        Console.OutputEncoding = Encoding.UTF8;
        Directory.CreateDirectory(Folder);

        double[][] X = Dane.Wczytaj();      // 101 próbek x 2 atrybuty
        Pisz($"Wczytano {X.Length} próbek, {X[0].Length} atrybuty. Ziarno losowania: {Ziarno}");
        Pisz();

        // ---- Raport 1: k-średnie, euklides, 3 grupy, 10 iteracji (stan po 4 i 10) ----
        Pisz("=== RAPORT 1 (k-średnie, miara euklidesowa, 3 grupy, 10 iteracji) ===");
        var r1 = KMeans(X, 3, 10, Euklides, new Random(Ziarno), new HashSet<int> { 4, 10 });
        foreach (var st in r1) RaportKMeans("Raport1 (k-srednie)", X, 3, st);

        // ---- Raport 2: k-średnie, tylko x1, 4 grupy, 10 iteracji ----
        Pisz("=== RAPORT 2 (k-średnie, odległość |x1_A - x1_B|, 4 grupy, 10 iteracji) ===");
        var r2 = KMeans(X, 4, 10, TylkoX1, new Random(Ziarno), new HashSet<int> { 4, 10 });
        foreach (var st in r2) RaportKMeans("Raport2 (k-srednie, tylko x1)", X, 4, st);

        // ---- Raport 3: FCM, euklides, 3 grupy, fcm_m = 2, 20 iteracji (stan po 4 i 20) ----
        Pisz("=== RAPORT 3 (FCM, miara euklidesowa, 3 grupy, fcm_m = 2, 20 iteracji) ===");
        var r3 = Fcm(X, 3, 20, 2.0, Euklides, new Random(Ziarno), new HashSet<int> { 4, 20 });
        foreach (var st in r3) RaportFcm("Raport3 (FCM)", X, 3, st);

        File.WriteAllText(Path.Combine(Folder, "raport.txt"), Raport.ToString(), Encoding.UTF8);
        Console.WriteLine($"Zapisano raport i wykresy w folderze: {Path.GetFullPath(Folder)}");
    }
}

// ------------------------- WYKRESY (czysty SVG) ------------------------------
static class Svg
{
    static readonly string[] Kolory = { "#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e" };

    public static void Zapisz(string sciezka, string tytul,
                              List<(string nazwa, List<double[]> pkt)> serie, double[][] srodki)
    {
        const double W = 820, H = 620, L = 70, R = 230, T = 50, B = 60;
        const double xmin = -2, xmax = 1.5, ymin = -2, ymax = 2.5;
        double pw = W - L - R, ph = H - T - B;
        double PX(double x) => L + (x - xmin) / (xmax - xmin) * pw;
        double PY(double y) => T + (ymax - y) / (ymax - ymin) * ph;
        string N(double v) => v.ToString("F1", CultureInfo.InvariantCulture);

        var sb = new StringBuilder();
        sb.AppendLine($"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{W}\" height=\"{H}\" font-family=\"sans-serif\" font-size=\"12\">");
        sb.AppendLine($"<rect width=\"{W}\" height=\"{H}\" fill=\"white\"/>");
        sb.AppendLine($"<text x=\"{N(L)}\" y=\"28\" font-size=\"16\" font-weight=\"bold\">{System.Security.SecurityElement.Escape(tytul)}</text>");
        sb.AppendLine($"<rect x=\"{N(L)}\" y=\"{N(T)}\" width=\"{N(pw)}\" height=\"{N(ph)}\" fill=\"none\" stroke=\"black\"/>");

        for (double x = xmin; x <= xmax + 1e-9; x += 0.5)
        {
            sb.AppendLine($"<line x1=\"{N(PX(x))}\" y1=\"{N(T)}\" x2=\"{N(PX(x))}\" y2=\"{N(T + ph)}\" stroke=\"#ddd\"/>");
            sb.AppendLine($"<text x=\"{N(PX(x))}\" y=\"{N(T + ph + 18)}\" text-anchor=\"middle\">{x.ToString("0.0", CultureInfo.InvariantCulture)}</text>");
        }
        for (double y = ymin; y <= ymax + 1e-9; y += 0.5)
        {
            sb.AppendLine($"<line x1=\"{N(L)}\" y1=\"{N(PY(y))}\" x2=\"{N(L + pw)}\" y2=\"{N(PY(y))}\" stroke=\"#ddd\"/>");
            sb.AppendLine($"<text x=\"{N(L - 8)}\" y=\"{N(PY(y) + 4)}\" text-anchor=\"end\">{y.ToString("0.0", CultureInfo.InvariantCulture)}</text>");
        }
        sb.AppendLine($"<text x=\"{N(L + pw / 2)}\" y=\"{N(H - 15)}\" text-anchor=\"middle\">x1</text>");
        sb.AppendLine($"<text x=\"20\" y=\"{N(T + ph / 2)}\" text-anchor=\"middle\" transform=\"rotate(-90 20 {N(T + ph / 2)})\">x2</text>");

        for (int k = 0; k < serie.Count; k++)
        {
            string kol = k < Kolory.Length ? Kolory[k] : "#888888";
            if (serie[k].nazwa.StartsWith("U<=")) kol = "#999999";
            foreach (var p in serie[k].pkt)
                sb.AppendLine($"<rect x=\"{N(PX(p[0]) - 4)}\" y=\"{N(PY(p[1]) - 4)}\" width=\"8\" height=\"8\" fill=\"none\" stroke=\"{kol}\" stroke-width=\"1.5\"/>");
            double ly = T + 20 + k * 22;
            sb.AppendLine($"<rect x=\"{N(L + pw + 20)}\" y=\"{N(ly - 8)}\" width=\"10\" height=\"10\" fill=\"none\" stroke=\"{kol}\" stroke-width=\"1.5\"/>");
            sb.AppendLine($"<text x=\"{N(L + pw + 38)}\" y=\"{N(ly + 1)}\">{System.Security.SecurityElement.Escape(serie[k].nazwa)}</text>");
        }

        foreach (var c in srodki)
            sb.AppendLine($"<circle cx=\"{N(PX(c[0]))}\" cy=\"{N(PY(c[1]))}\" r=\"6\" fill=\"black\" stroke=\"white\" stroke-width=\"1.5\"/>");
        double lyc = T + 20 + serie.Count * 22;
        sb.AppendLine($"<circle cx=\"{N(L + pw + 25)}\" cy=\"{N(lyc - 3)}\" r=\"5\" fill=\"black\"/>");
        sb.AppendLine($"<text x=\"{N(L + pw + 38)}\" y=\"{N(lyc + 1)}\">Środki grup</text>");

        sb.AppendLine("</svg>");
        File.WriteAllText(sciezka, sb.ToString(), Encoding.UTF8);
    }
}
