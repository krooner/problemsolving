#include <stdio.h>
int main(){
	char arr[1000000]; scanf("%s", arr);
	char cnt[26] = {0,};
	int idx = 0,  max = 0;
	while (1) {
		if (arr[idx] == '\0') break;
		int cidx = (int)arr[idx] - 97;
		if (cidx < 0){
			cnt[cidx+26] += 1;
			if (max < cnt[cidx+26])
				max = cnt[cidx+26];
		}
		else {
			cnt[cidx] += 1;
			if (max < cnt[cidx])
				max = cnt[cidx];
		}
		idx += 1;
	}
	
	idx = -1;
	for (int i = 0; i < 26; i++){
		if (cnt[i] == max){
			if (idx == -1)
				idx = i;
			else {
				printf("?");
				return 0;
			}
		}
	}

	printf("%c", idx+65);

	return 0;
}	
