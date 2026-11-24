"""(Re)generate the Visual Studio 2022 Python project files.

    python scripts/gen_vsproj.py

Writes PixelBridge.sln and PixelBridge.pyproj listing every .py file and the
main content files. Run it again before making the submission zip, so that
files added by the team appear in Visual Studio's Solution Explorer.
Startup file: main.py (no arguments -> GUI).
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_GUID = "{83EB2F30-561F-4250-8B70-A6957B989114}"
PYTHON_PROJECT_TYPE = "{888888A0-9F3D-457C-B088-3A5042F75D52}"
SKIP_DIRS = {".git", ".github", ".vs", "__pycache__", ".pytest_cache", "results", "env", "venv", ".venv"}
CONTENT_EXT = {".md", ".txt", ".ini"}


def collect():
    compile_files, content_files, folders = [], [], set()
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        rel_folder = os.path.relpath(folder, ROOT)
        for name in sorted(files):
            rel = os.path.normpath(os.path.join(rel_folder, name)).replace("/", "\\")
            ext = os.path.splitext(name)[1].lower()
            if ext == ".py":
                compile_files.append(rel)
            elif ext in CONTENT_EXT:
                content_files.append(rel)
            else:
                continue
            if rel_folder != ".":
                parts = rel_folder.replace("/", "\\").split("\\")
                for i in range(1, len(parts) + 1):
                    folders.add("\\".join(parts[:i]) + "\\")
    return compile_files, content_files, sorted(folders)


def pyproj(compile_files, content_files, folders) -> str:
    items = lambda tag, xs: "\n".join(f'    <{tag} Include="{x}" />' for x in xs)  # noqa: E731
    return f"""<Project DefaultTargets="Build" xmlns="http://schemas.microsoft.com/developer/msbuild/2003" ToolsVersion="4.0">
  <PropertyGroup>
    <Configuration Condition=" '$(Configuration)' == '' ">Debug</Configuration>
    <SchemaVersion>2.0</SchemaVersion>
    <ProjectGuid>{PROJECT_GUID}</ProjectGuid>
    <ProjectHome>.</ProjectHome>
    <StartupFile>main.py</StartupFile>
    <SearchPath>.</SearchPath>
    <WorkingDirectory>.</WorkingDirectory>
    <OutputPath>.</OutputPath>
    <Name>PixelBridge</Name>
    <RootNamespace>PixelBridge</RootNamespace>
    <LaunchProvider>Standard Python launcher</LaunchProvider>
    <EnableNativeCodeDebugging>False</EnableNativeCodeDebugging>
  </PropertyGroup>
  <PropertyGroup Condition=" '$(Configuration)' == 'Debug' ">
    <DebugSymbols>true</DebugSymbols>
    <EnableUnmanagedDebugging>false</EnableUnmanagedDebugging>
  </PropertyGroup>
  <PropertyGroup Condition=" '$(Configuration)' == 'Release' ">
    <DebugSymbols>true</DebugSymbols>
    <EnableUnmanagedDebugging>false</EnableUnmanagedDebugging>
  </PropertyGroup>
  <ItemGroup>
{items("Compile", compile_files)}
  </ItemGroup>
  <ItemGroup>
{items("Content", content_files)}
  </ItemGroup>
  <ItemGroup>
{items("Folder", folders)}
  </ItemGroup>
  <Import Project="$(MSBuildExtensionsPath32)\\Microsoft\\VisualStudio\\v$(VisualStudioVersion)\\Python Tools\\Microsoft.PythonTools.targets" />
</Project>
"""


def sln() -> str:
    g = PROJECT_GUID
    return f"""
Microsoft Visual Studio Solution File, Format Version 12.00
# Visual Studio Version 17
VisualStudioVersion = 17.0.31903.59
MinimumVisualStudioVersion = 10.0.40219.1
Project("{PYTHON_PROJECT_TYPE}") = "PixelBridge", "PixelBridge.pyproj", "{g}"
EndProject
Global
\tGlobalSection(SolutionConfigurationPlatforms) = preSolution
\t\tDebug|Any CPU = Debug|Any CPU
\t\tRelease|Any CPU = Release|Any CPU
\tEndGlobalSection
\tGlobalSection(ProjectConfigurationPlatforms) = postSolution
\t\t{g}.Debug|Any CPU.ActiveCfg = Debug|Any CPU
\t\t{g}.Release|Any CPU.ActiveCfg = Release|Any CPU
\tEndGlobalSection
\tGlobalSection(SolutionProperties) = preSolution
\t\tHideSolutionNode = FALSE
\tEndGlobalSection
EndGlobal
"""


def main() -> None:
    compile_files, content_files, folders = collect()
    with open(os.path.join(ROOT, "PixelBridge.pyproj"), "w", encoding="utf-8", newline="\r\n") as f:
        f.write(pyproj(compile_files, content_files, folders))
    with open(os.path.join(ROOT, "PixelBridge.sln"), "w", encoding="utf-8-sig", newline="\r\n") as f:
        f.write(sln())
    print(f"PixelBridge.pyproj: {len(compile_files)} .py files, {len(content_files)} content files")


if __name__ == "__main__":
    main()
