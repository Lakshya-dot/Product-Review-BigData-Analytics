Set-Location $PSScriptRoot
$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.InterfaceAlias -notmatch 'Loopback' -and $_.IPAddress -notlike '169.254.*'} | Select-Object -First 1).IPAddress
Write-Host "Starting ReviewSense Analytics Website..." -ForegroundColor Cyan
Write-Host "  Local URL:     http://localhost:8501" -ForegroundColor Green
if ($ip) { Write-Host "  Same-WiFi URL: http://$($ip):8501" -ForegroundColor Green }
& "C:\Users\laksh\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run dashboard\app.py --server.address 0.0.0.0 --server.port 8501
