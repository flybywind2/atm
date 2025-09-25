"""
FastAPI server startup script with enhanced configuration
"""

import uvicorn
import sys
import os
import socket
try:
    import psutil  # optional; used to list/kill processes on a port
except ImportError:
    psutil = None
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Import settings after adding to path
from app.config import settings


def is_port_available(host: str, port: int) -> bool:
    """포트가 사용 가능한지 확인"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.bind((host, port))
            return True
    except socket.error:
        return False


def find_processes_using_port(port: int) -> list:
    """특정 포트를 사용하는 프로세스들 찾기 (psutil 없으면 빈 목록 반환)"""
    processes = []
    if psutil is None:
        return processes
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                connections = proc.connections(kind='inet')
                for conn in connections:
                    if conn.laddr.port == port:
                        processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmdline': ' '.join(proc.info['cmdline'] or [])
                        })
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        print(f"프로세스 검색 중 오류: {e}")

    return processes


def find_available_port(host: str, start_port: int, max_attempts: int = 10) -> int:
    """사용 가능한 포트 찾기"""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(host, port):
            return port
    return None


if __name__ == "__main__":
    print("=== AI Problem Solving Copilot Server Startup ===")

    # 포트 사용 가능성 체크
    target_host = settings.SERVER_HOST
    target_port = settings.SERVER_PORT

    if not is_port_available(target_host, target_port):
        print(f"[ERROR] 포트 {target_port}이(가) 이미 사용 중입니다!")

        # 해당 포트를 사용하는 프로세스 찾기
        processes = find_processes_using_port(target_port)
        if processes:
            print("\n[INFO] 포트를 사용 중인 프로세스:")
            for proc in processes:
                print(f"  - PID: {proc['pid']}, Name: {proc['name']}")
                print(f"    Command: {proc['cmdline'][:100]}...")

            # 종료 옵션 제공
            if psutil is None:
                print("\n[WARN] psutil 미설치: 프로세스 종료는 건너뜁니다. 다른 포트로 시도합니다.")
                choice = 'n'
            else:
                choice = input(f"\n[PROMPT] 옵션을 선택하세요:\n  1) 기존 프로세스들을 종료하고 계속 (y)\n  2) 다른 포트 찾아서 실행 (n)\n  3) 종료 (q)\n선택 [y/n/q]: ").lower()

            if choice == 'y':
                if psutil is None:
                    print("[ERROR] psutil 미설치 상태로 종료를 수행할 수 없습니다.")
                else:
                    print("\n[INFO] 기존 프로세스들을 종료합니다...")
                    for proc in processes:
                        try:
                            psutil.Process(proc['pid']).terminate()
                            print(f"  [SUCCESS] PID {proc['pid']} 종료됨")
                        except Exception as e:
                            print(f"  [ERROR] PID {proc['pid']} 종료 실패: {e}")

                # 잠시 대기 후 다시 체크
                import time
                time.sleep(2)

                if psutil is not None and not is_port_available(target_host, target_port):
                    print(f"[WARNING] 포트 {target_port}이(가) 여전히 사용 중입니다. 강제 종료를 시도합니다...")
                    for proc in processes:
                        try:
                            psutil.Process(proc['pid']).kill()
                            print(f"  [SUCCESS] PID {proc['pid']} 강제 종료됨")
                        except Exception as e:
                            print(f"  [ERROR] PID {proc['pid']} 강제 종료 실패: {e}")
                    time.sleep(1)

            elif choice == 'n':
                # 사용 가능한 포트 찾기
                available_port = find_available_port(target_host, target_port + 1)
                if available_port:
                    print(f"[SUCCESS] 사용 가능한 포트를 찾았습니다: {available_port}")
                    target_port = available_port
                else:
                    print("[ERROR] 사용 가능한 포트를 찾을 수 없습니다.")
                    sys.exit(1)

            else:
                print("[INFO] 서버 시작이 취소되었습니다.")
                sys.exit(0)
        else:
            # psutil 미설치 또는 조회 실패로 프로세스 목록이 없을 때: 다른 포트 자동 탐색
            print("[INFO] 프로세스 목록을 확인할 수 없어 다른 포트를 탐색합니다...")
            available_port = find_available_port(target_host, target_port + 1)
            if available_port:
                print(f"[SUCCESS] 사용 가능한 포트를 찾았습니다: {available_port}")
                target_port = available_port
            else:
                print("[ERROR] 사용 가능한 포트를 찾을 수 없습니다.")
                sys.exit(1)

    print(f"\n[START] 서버를 시작합니다...")
    print(f"[INFO] 서버 주소: http://{target_host}:{target_port}")
    print(f"[INFO] API 문서: http://localhost:{target_port}/api/docs")
    print(f"[INFO] 대체 문서: http://localhost:{target_port}/api/redoc")
    print("\n[WARNING] 종료하려면 Ctrl+C를 누르세요")

    try:
        uvicorn.run(
            "app.main:app",
            host=target_host,
            port=target_port,
            reload=settings.RELOAD_MODE,
            log_level=settings.LOG_LEVEL,
            reload_dirs=[str(backend_dir)],
            workers=1  # Single worker for development
        )
    except KeyboardInterrupt:
        print("\n\n[INFO] 사용자에 의해 서버가 중지되었습니다.")
    except Exception as e:
        print(f"\n[ERROR] 서버 시작 실패: {e}")
        sys.exit(1)
