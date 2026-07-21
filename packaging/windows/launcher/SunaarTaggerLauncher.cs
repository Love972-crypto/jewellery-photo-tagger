using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

internal static class SunaarTaggerLauncher
{
    private const string AppName = "Sunaar Jewellery Tagger";

    [STAThread]
    private static void Main()
    {
        string root = AppDomain.CurrentDomain.BaseDirectory;
        string launcher = Path.Combine(root, "launcher", "Start-SunaarTagger.ps1");
        string setupMarker = Path.Combine(root, "app", ".venv", ".sunaar_requirements.sha256");

        if (!File.Exists(launcher))
        {
            ShowError("The launcher files are incomplete. Extract the complete ZIP again and keep every folder together.");
            return;
        }

        try
        {
            if (!File.Exists(setupMarker))
            {
                int setupExitCode = RunPowerShell(launcher, "-InstallOnly", true);
                if (setupExitCode != 0)
                {
                    ShowError(
                        "Automatic setup did not finish. Keep the internet connected and run " +
                        "'Install or Repair Dependencies.bat' once.");
                    return;
                }
            }

            RunPowerShell(launcher, string.Empty, false);
        }
        catch (Exception exception)
        {
            ShowError("The app could not start. " + exception.Message);
        }
    }

    private static int RunPowerShell(string script, string extraArguments, bool waitForExit)
    {
        string arguments = "-NoProfile -ExecutionPolicy Bypass -File \"" + script + "\"";
        if (!string.IsNullOrWhiteSpace(extraArguments))
        {
            arguments += " " + extraArguments;
        }

        Process process = Process.Start(new ProcessStartInfo
        {
            FileName = "powershell.exe",
            Arguments = arguments,
            WorkingDirectory = AppDomain.CurrentDomain.BaseDirectory,
            UseShellExecute = true
        });

        if (process == null)
        {
            throw new InvalidOperationException("Windows could not create the launcher process.");
        }

        if (!waitForExit)
        {
            return 0;
        }

        process.WaitForExit();
        return process.ExitCode;
    }

    private static void ShowError(string message)
    {
        MessageBox.Show(message, AppName, MessageBoxButtons.OK, MessageBoxIcon.Error);
    }
}
