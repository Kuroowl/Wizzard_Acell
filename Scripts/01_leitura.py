def main():
    parser = argparse.ArgumentParser(
        description="01_leitura: Mapeamento e consolidação dos ensaios T1, T2, T3..."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Caminho da pasta raiz selecionada no main",
    )
    args = parser.parse_args()
    raiz_path = Path(args.data_dir)

    try:
        # Carrega os dados brutos
        df_dados = mapear_e_carregar_dados(raiz_path)

        # 📁 Cria o diretório DadosTratados dentro da pasta selecionada pelo usuário
        output_dir = raiz_path / "DadosTratados"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 📄 Salva o arquivo com os dados carregados na pasta selecionada
        # Utilizando Pickle (.pkl) para não exigir a biblioteca pyarrow externa
        output_path = output_dir / "DadosTratados.pkl"
        df_dados.to_pickle(output_path)

        # (Caso prefira manter em .parquet, basta usar o trecho abaixo):
        # output_path = output_dir / "DadosTratados.parquet"
        # df_dados.to_parquet(output_path, index=False)

        print("\n✅ Leitura concluída com sucesso!")
        print("📊 Resumo do carregamento:")
        print(f"   - Total de registros/amostras: {len(df_dados)}")
        print(f"   - Sensores identificados: {df_dados['sensor'].unique().tolist()}")
        print(f"   - Condições/Ensaios mapeados: {sorted(df_dados['condicao'].unique().tolist())}")
        print(f"💾 Salvo em: {output_path.resolve()}\n")

    except Exception as e:
        print(f"\n❌ Erro na etapa 01_leitura: {e}")
        exit(1)