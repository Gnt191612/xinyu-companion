"""制作只更新桌面入口、保留已安装模型的一键修复包。"""
import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--desktop', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compiler', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    script = args.output / 'desktop-fix.iss'
    script.write_text(f'''[Setup]
AppId=XinyuDesktopFix201
AppName=心语桌面启动修复
AppVersion=2.0.1
DefaultDirName={{code:ExistingDir}}
UsePreviousAppDir=no
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
Uninstallable=no
CreateUninstallRegKey=no
OutputDir={args.output.resolve()}
OutputBaseFilename=Xinyu-Desktop-Fix-v2.0.1
SetupIconFile={args.project.resolve()}\\体验\\assets\\心语.ico
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes

[Languages]
Name: "chinesesimp"; MessagesFile: "compiler:Languages\\ChineseSimplified.isl"

[Files]
Source: "{args.desktop.resolve()}\\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{args.project.resolve()}\\体验\\desktop_edge.py"; DestDir: "{{app}}\\体验"; Flags: ignoreversion

[Code]
function ExistingDir(Param: String): String;
begin
  if not RegQueryStringValue(HKCU, 'Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\XinyuCompanionResearchV24_is1', 'InstallLocation', Result) then
    Result := ExpandConstant('{{localappdata}}\\Programs\\XinyuCompanion');
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
    if not FileExists(ExpandConstant('{{app}}\\.venv\\Scripts\\python.exe')) or
       not FileExists(ExpandConstant('{{app}}\\体验\\开场.html')) or
       not FileExists(ExpandConstant('{{app}}\\体验\\chat_app.py')) then
    begin
      Result := '请选择已安装的心语目录。本修复包不包含模型，需先安装完整离线包。';
    end;
end;
''', encoding='utf-8-sig')
    subprocess.run([str(args.compiler), str(script)], check=True)


if __name__ == '__main__':
    main()
