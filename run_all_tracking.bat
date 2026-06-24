@echo off
echo Starting tracking for goc_cheo...
ev\Scripts\python tools/track_and_label.py --prefix goc_cheo --start 1 --end 190 --api-key YOUR_OPENAI_API_KEY --interval 2

echo.
echo Starting tracking for goc_thang_phai (Part 1)...
ev\Scripts\python tools/track_and_label.py --prefix goc_thang_phai --start 1 --end 220 --api-key YOUR_OPENAI_API_KEY --interval 2

echo.
echo Starting tracking for goc_thang_phai (Part 2)...
ev\Scripts\python tools/track_and_label.py --prefix goc_thang_phai --start 221 --end 244 --api-key YOUR_OPENAI_API_KEY --interval 2

echo.
echo Starting tracking for goc_thang_trai...
ev\Scripts\python tools/track_and_label.py --prefix goc_thang_trai --start 1 --end 242 --api-key YOUR_OPENAI_API_KEY --interval 2

echo.
echo All tracking finished!
