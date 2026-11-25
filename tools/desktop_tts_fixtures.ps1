param([Parameter(Mandatory=$true)][string]$Output,
      [ValidateSet('zh','en')][string]$Language='zh',
      [string]$Manifest)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$zh = $null
$en = $null
if ($Language -eq 'zh') {
    Add-Type -AssemblyName System.Runtime.WindowsRuntime
    $null = [Windows.Media.SpeechSynthesis.SpeechSynthesizer,Windows.Media.SpeechSynthesis,ContentType=WindowsRuntime]
    $streamType = [Windows.Media.SpeechSynthesis.SpeechSynthesisStream,Windows.Media.SpeechSynthesis,ContentType=WindowsRuntime]
    $asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
        $_.Name -eq 'AsTask' -and $_.IsGenericMethodDefinition -and $_.GetParameters().Length -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
    } | Select-Object -First 1
    $taskMethod = $asTask.MakeGenericMethod($streamType)
    $zh = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::new()
} else {
    Add-Type -AssemblyName System.Speech
    $en = [System.Speech.Synthesis.SpeechSynthesizer]::new()
}
$outDir = [IO.Path]::GetFullPath($Output)
[IO.Directory]::CreateDirectory($outDir) | Out-Null
$fixtures = @(
    @{name='zh-meeting';language='zh';voice='Microsoft Huihui';text='今天下午三点开会，请带上会议记录。'},
    @{name='zh-check';language='zh';voice='Microsoft Kangkang';text='请检查录音文件，确认没有丢失任何一段。'},
    @{name='zh-save';language='zh';voice='Microsoft Yaoyao';text='我们先测试麦克风，再把转写结果保存到电脑。'},
    @{name='en-meeting';language='en';voice='Microsoft Zira Desktop';text='The meeting starts at three. Please bring your notes.'},
    @{name='en-save';language='en';voice='Microsoft Zira Desktop';text='Check the microphone and save the recording.'}
)
if ($Manifest) {
    $fixtures = [IO.File]::ReadAllText([IO.Path]::GetFullPath($Manifest), [Text.Encoding]::UTF8) | ConvertFrom-Json
}
try {
    foreach ($f in ($fixtures | Where-Object language -eq $Language)) {
        $path = Join-Path $outDir ($f.name + '.wav')
        if ($f.language -eq 'zh') {
            $zh.Voice = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices |
                Where-Object DisplayName -eq $f.voice | Select-Object -First 1
            $operation = $zh.SynthesizeTextToStreamAsync($f.text)
            $task = $taskMethod.Invoke($null, @($operation))
            $task.GetAwaiter().GetResult() | Out-Null
            $input = [System.IO.WindowsRuntimeStreamExtensions]::AsStreamForRead($task.Result)
            $stream = [IO.File]::Create($path)
            try { $input.CopyTo($stream) } finally { $input.Dispose(); $stream.Dispose() }
        } else {
            if ($en.Voice.Name -ne $f.voice) { $en.SelectVoice($f.voice) }
            $en.SetOutputToWaveFile($path)
            $en.Speak($f.text)
            $en.SetOutputToNull()
        }
    }
    $fixtures | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $outDir 'fixtures.json')
} finally {
    if ($zh) { $zh.Dispose() }
    if ($en) { $en.Dispose() }
}
