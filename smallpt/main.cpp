extern int original(int argc, char *argv[]);
extern int sycl_gtx(int argc, char *argv[]);

int main(int argc, char *argv[]) {
	original(argc, argv);
	return 0;
}
