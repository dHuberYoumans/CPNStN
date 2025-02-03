# CPNStN

## Getting Started

### GNU Screen

1. Download the source and extract 
```
$ wget http://git.savannah.gnu.org/cgit/screen.git/snapshot/v.4.3.1.tar.gz
$ tar -xvf v.4.3.1.tar.gz
$ cd v.4.3.1/src/
```

2. Build GNU Screen
```
$ ./autogen.sh
$ ./configure
$ make
```

3. Run GNU Screen
``` 
screen -S <session_name>
```
