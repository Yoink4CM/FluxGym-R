# FluxGym-R
FluxGym modernized + ROCm support.

Hi there 👋

Feel free to buy us a coffee to help support development!

https://buymeacoffee.com/yoink4cm

FluxGym-R stems from the original FluxGym project: https://fluxgym.org/

We tried to modernize the UI for slightly better flow, and of course, the biggie, add ROCm support!  We've also added another model for better captioning.

ROCm 7.2 and 10 are integrated, although if you're reading this sentence, we haven't tested 10 yet.  We've also (theoretically) added support for CU128 (5000 series cards from Nvidia).  Again, currently untested.

We will make tweaks as needed if the community reports issues.

***************
How to install


Your system must have ROCm or CUDA installed.  

git clone https://github.com/Yoink4CM/FluxGym-R
cd FluxGym-R
./install.sh
./app-launch.sh

Once loaded browse to http://local_ip:7680
