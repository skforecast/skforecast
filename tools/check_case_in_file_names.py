"""
check_case_in_file_names.py

Script para detectar archivos con convenciones de nombres incorrectas.
Busca archivos que deberían estar en minúsculas pero contienen mayúsculas
no semánticas (excluyendo casos válidos como _X, _Y para variables).

Uso:
    python tools/check_case_in_file_names.py
"""
import sys
from pathlib import Path


def check_incorrect_case_in_filenames():
    """
    Busca archivos con mayúsculas incorrectas en el directorio skforecast.
    
    Returns
    -------
    issues : list
        Lista de rutas relativas de archivos con problemas.
    
    """
    # Obtener la ruta raíz del repositorio (padre del directorio tools/)
    repo_root = Path(__file__).resolve().parent.parent
    skforecast_dir = repo_root / "skforecast"

    if not skforecast_dir.exists():
        print(f"❌ Error: No se encontró el directorio {skforecast_dir}")
        return None

    # Patrones que son incorrectos (no semánticos)
    incorrect_patterns = [
        'fixtures_',  # fixtures debería ser todo minúsculas
        'conftest',   # conftest debería ser todo minúsculas
    ]

    # Excepciones válidas (mayúsculas con significado semántico)
    valid_uppercase = ['_X', '_Y']

    issues = []
    for pattern in incorrect_patterns:
        for path in skforecast_dir.rglob(f"*{pattern}*.py"):
            filename = path.name
            if any(c.isupper() for c in filename.replace('.py', '')):
                # Verificar que no sea por razón semántica válida
                if not any(valid in filename for valid in valid_uppercase):
                    # Guardar ruta relativa para mejor legibilidad
                    relative_path = path.relative_to(repo_root)
                    issues.append(str(relative_path))

    return issues


def main():
    """Función principal del script."""
    print("=== Buscando archivos con mayúsculas INCORRECTAS ===\n")

    issues = check_incorrect_case_in_filenames()

    if issues is None:
        sys.exit(1)

    if issues:
        print("⚠️  Archivos con mayúsculas INCORRECTAS:\n")
        for issue in issues:
            print(f"  {issue}")
        sys.exit(1)
    else:
        print("✅ No se encontraron problemas")
        sys.exit(0)


if __name__ == "__main__":
    main()
