# Kit de Claude Code para ags-hydroscope

**Importante:** 17 de los 20 archivos están en las carpetas `.claude/` y `.github/`. Como empiezan con
punto, macOS, Windows y Linux las ocultan. Para verlas: `ls -la` en la terminal, Cmd+Shift+. en Finder,
Ctrl+H en Linux o Vista → Mostrar → Elementos ocultos en Windows.

La forma más segura de instalar es el script que viene junto a esta carpeta:

```bash
python instalar_kit_claude.py RUTA_DEL_REPO
```

También puedes copiar todo el contenido de esta carpeta (incluidas `.claude/` y `.github/`) a la raíz
del repositorio y hacer commit. Las instrucciones completas están en la sección
"Claude Code y Claude Desktop" del plan del proyecto.

| Archivo | Para qué sirve |
| --- | --- |
| `CLAUDE.md` | Contexto que Claude lee en cada sesión: comandos, estructura, invariantes del protocolo, reglas duras |
| `.claude/settings.json` | Permisos compartidos (qué se permite, qué pregunta, qué se prohíbe) y el hook de formato |
| `.claude/hooks/format-python.sh` | Aplica `ruff format` y `ruff check --fix` a cada archivo .py que Claude edita |
| `.claude/agents/*.md` | 6 subagentes: ml-engineer, geo-data-engineer, experiment-auditor, paper-writer, api-builder, frontend-builder |
| `.claude/skills/*/SKILL.md` | 6 comandos: /bootstrap-repo, /new-experiment, /results-tables, /protocol-check, /sprint-status, /paper-section |
| `.claude/rules/*.md` | Reglas que solo se cargan al tocar `paper/`, `slides/` o `frontend/` |
| `.github/workflows/claude.yml` | Permite escribir `@claude` en issues y PR de GitHub |
| `PROMPTS.md` | Prompts listos por tarea y sprint |

Pasos mínimos:

```bash
# 1. Instalar Claude Code (macOS, Linux o WSL)
curl -fsSL https://claude.ai/install.sh | bash
# Windows PowerShell: irm https://claude.ai/install.ps1 | iex

# 2. Copiar el kit al repo y dar permiso de ejecución al hook
cp -r ags-hydroscope-claude-kit/. ags-hydroscope/
chmod +x ags-hydroscope/.claude/hooks/format-python.sh

# 3. Abrir Claude Code en el repo e iniciar sesión con tu cuenta Pro/Max/Team
cd ags-hydroscope && claude

# 4. Dentro de Claude Code
/bootstrap-repo
```

El hook necesita `jq` y `ruff` en el PATH (`ruff` viene en environment.yml; `jq` se instala con el
gestor de paquetes del sistema).
